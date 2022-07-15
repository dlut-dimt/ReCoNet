from tabulate import tabulate
from torch import nn


def pretty_vars(module: nn.Module) -> str:
    table = [[n, p.mean(), p.grad.mean()] for n, p in module.named_parameters() if p.grad is not None]
    return tabulate(table, headers=['layer', 'weights', 'grad'], tablefmt='pretty')
