"""DDP HessianOperator: 2-rank gloo CPU test that the all-reduced HVP equals the single-process HVP."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from hessian_eigenthings.operators import HessianOperator
from hessian_eigenthings.operators.distributed import DDPHessianOperator


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _build_model_and_data(dtype: torch.dtype = torch.float64):
    g = torch.Generator()
    g.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2)).to(dtype)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=dtype)

    g2 = torch.Generator()
    g2.manual_seed(1)
    x = torch.randn(8, 3, generator=g2, dtype=dtype)
    y = torch.randn(8, 2, generator=g2, dtype=dtype)
    return model, x, y


def _supervised_loss(model: nn.Module, batch) -> torch.Tensor:
    x, y = batch
    return nn.functional.mse_loss(model(x), y)


def _ddp_worker(rank: int, world_size: int, port: int, vec: torch.Tensor, out_path: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        model, x, y = _build_model_and_data()
        per_rank = x.shape[0] // world_size
        x_local = x[rank * per_rank : (rank + 1) * per_rank]
        y_local = y[rank * per_rank : (rank + 1) * per_rank]
        op = DDPHessianOperator(
            model=model, dataloader=[(x_local, y_local)], loss_fn=_supervised_loss
        )
        result = op.matvec(vec)
        if rank == 0:
            torch.save(result, out_path)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_ddp_hvp_matches_single_process(tmp_path) -> None:
    """2-rank DDP HVP on split data == single-process HVP on the union, modulo the avg/sum scaling."""
    model, x, y = _build_model_and_data()
    op_single = HessianOperator(model=model, dataloader=[(x, y)], loss_fn=_supervised_loss)

    g = torch.Generator()
    g.manual_seed(2)
    v = torch.randn(op_single.size, generator=g, dtype=torch.float64)
    expected = op_single.matvec(v)

    out = tmp_path / "rank0.pt"
    port = _free_port()
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(2):
        p = ctx.Process(target=_ddp_worker, args=(rank, 2, port, v, str(out)))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=120)
    for p in procs:
        assert p.exitcode == 0, f"DDP worker exited with code {p.exitcode}"

    ddp_result = torch.load(out, weights_only=True)
    # DDP all-reduce-mean: each rank's HVP averaged over its half-batch (4 samples each
    # via mse_loss reduction='mean'); cross-rank average of those equals the
    # single-process HVP on the full 8 samples (also mean-reduced).
    torch.testing.assert_close(ddp_result, expected, rtol=1e-7, atol=1e-9)
