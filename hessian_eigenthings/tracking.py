"""
This module enables tracking the hessian throughout training by incrementally
updating eigenvalue/eigenvec estimates as the model progresses.
"""
from .hvp_operator import HVPOperator, deflated_power_iteration
from .power_iter import LambdaOperator, power_iteration


class HessianTracker:
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
        self.deflated_power_fn = lambda op: deflated_power_iteration(op,
                                                                     num_eigenthigns=num_eigenthings,
                                                                     power_iter_steps=power_iter_steps,
                                                                     power_iter_err_threshold=power_iter_err_threshold,
                                                                     momentum=momentum,
                                                                     use_gpu=use_gpu)
        self.power_iter_fn = lambda op, prev: power_iteration(op,
                                                              steps=power_iter_steps,
                                                              error_threshold=power_iter_err_threshold,
                                                              momentum=momentum,
                                                              use_gpu=use_gpu,
                                                              init_vec=prev)
        # Set initial eigenvalue estimates
        self.eigenvecs = None
        self.eigenvals = None

    def step(self):
        """
        Perform power iteration, starting from the initial eigen estimates
        we accrued from the previous steps.
        """
        if self.eigenvals is None:
            self.eigenvals, self.eigenvecs = self.deflated_power_fn(self.hvp_operator)
            return

        # Update existing estimates, one at a time.
        def _deflate(x, val, vec):
            return val * vec.dot(x) * vec

        current_op = self.hvp_operator
        for i in range(self.num_eigenthings):
            prev_eigenvec = self.eigenvecs[i]
            new_eigenval, new_eigenvec = self.power_iter_fn(current_op, prev_eigenvec)

            def _new_op_fn(x, op=current_op, val=new_eigenval, vec=new_eigenvec):
                return op.apply(x) - _deflate(x, val, vec)
            current_op = LambdaOperator(_new_op_fn, self.hvp_operator.size)
            self.eigenvals[i] = new_eigenval
            self.eigenvecs[i] = new_eigenvec

    def get_eigenthings(self):
        return self.eigenvals, self.eigenvecs


