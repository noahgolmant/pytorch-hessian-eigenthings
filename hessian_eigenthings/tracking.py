"""
This module enables tracking the hessian throughout training by incrementally
updating eigenvalue/eigenvec estimates as the model progresses.
"""
from .hvp_operator import HVPOperator
from .power_iter import LambdaOperator, power_iteration, deflated_power_iteration


class HessianTracker:
    """
    This class incrementally tracks the top `num_eigenthings` eigenval/vec
    pairs for the hessian of the loss of the given model. It uses
    accelerated stochastic power iteration with deflation to do this.

    model: PyTorch model for which we want to track the hessian
    dataloader: PyTorch dataloader that lets us compute the gradient
    loss: objective function that we take the hessian of
    num_eigenthings: number of eigenval/vec pairs to track
    power_iter_steps: default number of power iteration steps for each deflation step
    power_iter_err_threshold: error tolerance for early stopping in power iteration
    momentum: acceleration term for accelerated stochastic power iteration
    max_samples: max number of samples we can compute the grad of at once
    use_gpu: use cuda or not
    """
    def __init__(self, model, dataloader, loss,
                 num_eigenthings=10,
                 power_iter_steps=20,
                 power_iter_err_threshold=1e-4,
                 momentum=0.0,
                 max_samples=512,
                 use_gpu=True):
        self.num_eigenthings = num_eigenthings
        self.hvp_operator = HVPOperator(model, dataloader, loss,
                                        use_gpu=use_gpu, max_samples=max_samples)

        # This function computes the initial eigenthing estimates
        def _deflated_power_fn(op):
            return deflated_power_iteration(op,
                                            num_eigenthings,
                                            power_iter_steps,
                                            power_iter_err_threshold,
                                            momentum,
                                            use_gpu)
        self.deflated_power_fn = _deflated_power_fn

        # This function will update a single vector using the deflated op
        def _power_iter_fn(op, prev, steps):
            return power_iteration(op,
                                   steps,
                                   power_iter_err_threshold,
                                   momentum,
                                   prev,
                                   use_gpu)
        self.power_iter_fn = _power_iter_fn

        # Set initial eigenvalue estimates
        self.eigenvecs = None
        self.eigenvals = None

    def step(self, power_iter_steps=None):
        """
        Perform power iteration, starting from the initial eigen estimates
        we accrued from the previous steps.
        """
        # Take the first estimate if we need to.
        if self.eigenvals is None:
            self.eigenvals, self.eigenvecs = self.deflated_power_fn(self.hvp_operator)
            return

        # Allow a variable number of update steps during training.
        if power_iter_steps is None:
            power_iter_steps = self.power_iter_steps

        # Update existing estimates, one at a time.
        def _deflate(x, val, vec):
            return val * vec.dot(x) * vec

        current_op = self.hvp_operator
        for i in range(self.num_eigenthings):
            prev_eigenvec = self.eigenvecs[i]
            # Use the previous eigenvec estimate as the starting point.
            new_eigenval, new_eigenvec = self.power_iter_fn(current_op, prev_eigenvec,
                                                            power_iter_steps)

            # Deflate the HVP operator using this new estimate.
            def _new_op_fn(x, op=current_op, val=new_eigenval, vec=new_eigenvec):
                return op.apply(x) - _deflate(x, val, vec)
            current_op = LambdaOperator(_new_op_fn, self.hvp_operator.size)
            self.eigenvals[i] = new_eigenval
            self.eigenvecs[i] = new_eigenvec

    def get_eigenthings(self):
        """ Get current estimate of the eigenthings """
        return self.eigenvals, self.eigenvecs


