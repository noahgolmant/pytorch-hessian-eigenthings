""" Basic linear operator abstractions """


class Operator:
    """
    maps x -> Lx for a linear operator L
    """

    def __init__(self, size):
        self.size = size

    def apply(self, vec):
        """
        Function mapping vec -> L vec where L is a linear operator
        """
        raise NotImplementedError


class LambdaOperator(Operator):
    """
    Linear operator based on a provided lambda function
    """

    def __init__(self, apply_fn, size):
        super(LambdaOperator, self).__init__(size)
        self.apply_fn = apply_fn

    def apply(self, x):
        return self.apply_fn(x)
