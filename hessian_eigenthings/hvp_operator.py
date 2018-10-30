"""
This module defines a linear operator to compute the hessian-vector product
for a given pytorch model using subsampled data.
"""
import torch
from .power_iter import Operator, deflated_power_iteration


class HVPOperator(Operator):
    """
    Use PyTorch autograd for Hessian Vec product calculation
    model:  PyTorch network to compute hessian for
    dataloader: pytorch dataloader that we get examples from to compute grads
    loss:   Loss function to descend (e.g. F.cross_entropy)
    use_gpu: use cuda or not
    """

    def __init__(self, model, dataloader, criterion, use_gpu=True):
        size = int(sum(p.numel() for p in model.parameters()))
        super(HVPOperator, self).__init__(size)
        self.grad_vec = torch.zeros(size)
        self.model = model
        if use_gpu:
            self.model = self.model.cuda()
        self.dataloader = dataloader
        # Make a copy since we will go over it a bunch
        self.dataloader_iter = iter(dataloader)
        self.criterion = criterion
        self.use_gpu = use_gpu

    def apply(self, vec):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        # compute original gradient, tracking computation graph
        self.zero_grad()
        grad_vec = self.prepare_grad()
        # compute the product
        grad_product = torch.sum(grad_vec * vec)
        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(grad_product, self.model.parameters())
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat([g.contiguous().view(-1)
                                      for g in grad_grad])
        return hessian_vec_prod

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def prepare_grad(self):
        """
        Compute gradient w.r.t loss over all parameters and vectorize
        """
        try:
            input, target = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            input, target = next(self.dataloader_iter)

        if self.use_gpu:
            input = input.cuda()
            target = target.cuda()

        output = self.model(input)
        loss = self.criterion(output, target)
        grad_dict = torch.autograd.grad(
            loss, self.model.parameters(), create_graph=True)
        self.grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
        return self.grad_vec


def compute_hessian_eigenthings(model, dataloader, loss,
                                num_eigenthings=10,
                                power_iter_steps=20,
                                power_iter_err_threshold=1e-4,
                                use_gpu=True):
    """
    Computes the top `num_eigenthings` eigenvalues and eigenvecs
    for the hessian of the given model by using subsampled power iteration
    with deflation and the hessian-vector product
    """
    hvp_operator = HVPOperator(model, dataloader, loss, use_gpu=use_gpu)
    eigenvals, eigenvecs = deflated_power_iteration(hvp_operator,
                                                    num_eigenthings,
                                                    power_iter_steps,
                                                    power_iter_err_threshold,
                                                    use_gpu=use_gpu)
    return eigenvals, eigenvecs
