# Distributed (DDP)

`DDPHessianOperator` averages the Hessian-vector product across `torch.distributed` ranks. Each rank receives its own shard of the dataset (typical pattern: `DistributedSampler`); the per-rank HVP is all-reduced before being returned.

## Setup

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from hessian_eigenthings.algorithms import lanczos
from hessian_eigenthings.loss_fns import supervised_loss
from hessian_eigenthings.operators.distributed import DDPHessianOperator


def main(rank: int, world_size: int) -> None:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = build_model().cuda(rank)
    model = DDP(model, device_ids=[rank])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)

    op = DDPHessianOperator(
        model=model,
        dataloader=loader,
        loss_fn=supervised_loss(criterion),
    )
    result = lanczos(op, k=5, max_iter=30, seed=0)
    if rank == 0:
        print(result.eigenvalues)

    dist.destroy_process_group()
```

Launch with `torchrun --nproc_per_node=N your_script.py` or `torch.multiprocessing.spawn`.

## Why this is necessary

DDP's gradient hooks fire on `.grad` accumulation during `loss.backward()`. Our HVP uses `torch.autograd.grad(...)` directly, which does **not** trigger those hooks. So a plain `HessianOperator` wrapping a DDP-wrapped model would compute the per-rank HVP without averaging — each rank would see a different result.

`DDPHessianOperator` adds an explicit `torch.distributed.nn.all_reduce` (autograd-aware) on the gradients before the second backward, and on the final HVP, so all ranks see the same answer.

## Determinism

The dataloader must yield deterministic batches per rank — set `shuffle=False` on the sampler, no random data augmentation that varies between matvecs. Otherwise the same input vector $v$ produces different $Hv$ across calls (because the underlying dataset shuffles), breaking iterative algorithms.

## Finite-difference is also supported

`DDPHessianOperator(..., method="finite_difference")` works the same way — the FD path's `g_plus` and `g_minus` are each all-reduced separately. This is the recommended path for large LLMs where double-backward is impractical.

## Limitations

- This is the data-parallel case only. **FSDP** (sharded weights) is on the v1.1 roadmap and uses the finite-difference HVP path described in Granziol & Juarev 2026.
- The eigenvector $v$ is duplicated on every rank. For models too large to fit a parameter-shaped vector on a single rank, you'll need the sharded backend (also v1.1).
- Only a single process group is used; `process_group=` may be supplied to the constructor for nested parallelism setups.
