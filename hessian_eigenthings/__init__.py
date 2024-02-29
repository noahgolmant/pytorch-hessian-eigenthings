""" Top-level module for hessian eigenvec computation """
from hessian_eigenthings.power_iter import power_iteration, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from hessian_eigenthings.hvp_operator import HVPOperator
from hessian_eigenthings.utils import Device

name = "hessian_eigenthings"


def compute_hessian_eigenthings(
    model,
    dataloader,
    loss,
    num_eigenthings=10,
    full_dataset=True,
    mode="power_iter",
    device: Device = "cuda",
    fp16=False,
    max_possible_gpu_samples=2**16,
    **kwargs
):
    """
    Computes the top `num_eigenthings` eigenvalues and eigenvecs
    for the hessian of the given model by using subsampled power iteration
    with deflation and the hessian-vector product

    Parameters
    ---------------

    model : Module
        pytorch model for this netowrk
    dataloader : torch.data.DataLoader
        dataloader with x,y pairs for which we compute the loss.
    loss : torch.nn.modules.Loss | torch.nn.functional criterion
        loss function to differentiate through
    num_eigenthings : int
        number of eigenvalues/eigenvecs to compute. computed in order of
        decreasing eigenvalue magnitude.
    full_dataset : boolean
        if true, each power iteration call evaluates the gradient over the
        whole dataset.
        (if False, you might want to check if the eigenvalue estimate variance
         depends on batch size)
    mode : str ['power_iter', 'lanczos']
        which backend algorithm to use to compute the top eigenvalues.
    use_gpu:
        if true, attempt to use cuda for all lin alg computatoins
    fp16: bool
        if true, store and do math with eigenvectors, gradients, etc. in fp16.
        (you should test if this is numerically stable for your application)
    max_possible_gpu_samples:
        the maximum number of samples that can fit on-memory. used
        to accumulate gradients for large batches.
        (note: if smaller than dataloader batch size, this can have odd
         interactions with batch norm statistics)
    **kwargs:
        contains additional parameters passed onto lanczos or power_iter.
    """
    hvp_operator = HVPOperator(
        model,
        dataloader,
        loss,
        device=device,
        full_dataset=full_dataset,
        max_possible_gpu_samples=max_possible_gpu_samples,
    )
    eigenvals, eigenvecs = None, None
    if mode == "power_iter":
        eigenvals, eigenvecs = deflated_power_iteration(
            hvp_operator, num_eigenthings, device=device, fp16=fp16, **kwargs
        )
    elif mode == "lanczos":
        eigenvals, eigenvecs = lanczos(
            hvp_operator, num_eigenthings, device=device, fp16=fp16, **kwargs
        )
    else:
        raise ValueError("Unsupported mode %s (must be power_iter or lanczos)" % mode)
    return eigenvals, eigenvecs


__all__ = [
    "power_iteration",
    "deflated_power_iteration",
    "lanczos",
    "HVPOperator",
    "compute_hessian_eigenthings",
]
