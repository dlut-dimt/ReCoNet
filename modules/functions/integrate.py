from torch import Tensor

from modules.functions.transformer import transformer


def integrate(n_step: int, flow: Tensor) -> Tensor:
    scale = 1.0 / (2 ** n_step)
    flow = flow * scale
    for _ in range(n_step):
        i_flow = flow.permute(0, 3, 1, 2)  # [b, 2, h, w]
        o_flow = transformer(i_in=i_flow, flow=flow)[0].permute(0, 2, 3, 1)  # [b, h, w, 2]
        flow = flow + o_flow
    return flow
