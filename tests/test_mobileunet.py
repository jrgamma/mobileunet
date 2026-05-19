from typing import Callable

import torch
import torch.nn as nn

from mobileunet import mobileunet

import pytest


@pytest.mark.parametrize(
    "model_factory",
    [
        mobileunet.mobileunet_small,
        mobileunet.mobileunet_large,
    ],
)
def test_model_forward(model_factory: Callable[..., nn.Module]):
    B, C, H, W = 7, 3, 360, 640
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_factory().to(device=device)
    test_input = torch.randn(B, C, H, W, device=device)
    test_output = model(test_input)

    assert tuple(test_output.shape) == (B, 1, H, W)
