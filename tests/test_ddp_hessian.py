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


def _ddp_worker(
    rank: int,
    world_size: int,
    port: int,
    vec: torch.Tensor,
    out_path: str,
    method: str = "autograd",
    fd_eps: float | None = None,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        model, x, y = _build_model_and_data()
        per_rank = x.shape[0] // world_size
        x_local = x[rank * per_rank : (rank + 1) * per_rank]
        y_local = y[rank * per_rank : (rank + 1) * per_rank]
        kwargs: dict = {"method": method}
        if fd_eps is not None:
            kwargs["fd_eps"] = fd_eps
        op = DDPHessianOperator(
            model=model, dataloader=[(x_local, y_local)], loss_fn=_supervised_loss, **kwargs
        )
        result = op.matvec(vec)
        if rank == 0:
            torch.save(result, out_path)
    finally:
        dist.destroy_process_group()


def _run_ddp_and_get_rank0(
    tmp_path,
    vec: torch.Tensor,
    method: str,
    fd_eps: float | None = None,
) -> torch.Tensor:
    out = tmp_path / "rank0.pt"
    port = _free_port()
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(2):
        p = ctx.Process(target=_ddp_worker, args=(rank, 2, port, vec, str(out), method, fd_eps))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=120)
    for p in procs:
        assert p.exitcode == 0, f"DDP worker exited with code {p.exitcode}"
    return torch.load(out, weights_only=True)


@pytest.mark.distributed
def test_ddp_hvp_autograd_matches_single_process(tmp_path) -> None:
    """2-rank DDP HVP on split data == single-process HVP on the union, modulo the avg/sum scaling."""
    model, x, y = _build_model_and_data()
    op_single = HessianOperator(model=model, dataloader=[(x, y)], loss_fn=_supervised_loss)

    g = torch.Generator()
    g.manual_seed(2)
    v = torch.randn(op_single.size, generator=g, dtype=torch.float64)
    expected = op_single.matvec(v)

    ddp_result = _run_ddp_and_get_rank0(tmp_path, v, method="autograd")
    # DDP all-reduce-mean: each rank's HVP averaged over its half-batch (4 samples each
    # via mse_loss reduction='mean'); cross-rank average of those equals the
    # single-process HVP on the full 8 samples (also mean-reduced).
    torch.testing.assert_close(ddp_result, expected, rtol=1e-7, atol=1e-9)


@pytest.mark.distributed
def test_ddp_hvp_finite_difference_matches_single_process(tmp_path) -> None:
    """Same correctness check for the FD HVP path: 2-rank DDP FD == single-process FD."""
    model, x, y = _build_model_and_data()
    fd_eps = 1e-5
    op_single_fd = HessianOperator(
        model=model,
        dataloader=[(x, y)],
        loss_fn=_supervised_loss,
        method="finite_difference",
        fd_eps=fd_eps,
    )

    g = torch.Generator()
    g.manual_seed(3)
    v = torch.randn(op_single_fd.size, generator=g, dtype=torch.float64)
    expected_fd = op_single_fd.matvec(v)

    ddp_result = _run_ddp_and_get_rank0(tmp_path, v, method="finite_difference", fd_eps=fd_eps)
    # Same eps on both sides, so the FD bias cancels out and only the remaining
    # discrepancy is floating-point cancellation noise (~1e-9 at fp64 + eps=1e-5).
    torch.testing.assert_close(ddp_result, expected_fd, rtol=1e-7, atol=1e-9)


@pytest.mark.distributed
def test_ddp_finite_difference_close_to_autograd(tmp_path) -> None:
    """2-rank DDP FD HVP should match 2-rank DDP autograd HVP within the predicted FD bias."""
    g = torch.Generator()
    g.manual_seed(4)
    model, x, y = _build_model_and_data()
    op_single = HessianOperator(model=model, dataloader=[(x, y)], loss_fn=_supervised_loss)
    v = torch.randn(op_single.size, generator=g, dtype=torch.float64)

    autograd_ddp = _run_ddp_and_get_rank0(tmp_path, v, method="autograd")
    fd_ddp = _run_ddp_and_get_rank0(tmp_path, v, method="finite_difference", fd_eps=1e-5)

    rel = (fd_ddp - autograd_ddp).norm() / autograd_ddp.norm().clamp(min=1e-12)
    assert rel.item() < 1e-6  # O(eps^2) bias bound at fp64 + eps=1e-5
