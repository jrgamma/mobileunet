import copy
import dataclasses
import itertools
import functools
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import feature_extraction, mobilenetv3, WeightsEnum
from torchvision.ops import misc


class _ConvTranspose2dNormActivation(misc.ConvNormActivation):
    """Conv transpose analog of ConvNormActivation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...] = 3,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] | str | None = None,
        groups: int = 1,
        norm_layer: Callable[..., nn.Module] | None = torch.nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = torch.nn.ReLU,
        dilation: int | tuple[int, ...] = 1,
        inplace: bool | None = True,
        bias: bool | None = None,
    ) -> None:
        conv_layer = functools.partial(nn.ConvTranspose2d, output_padding=1)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            conv_layer,
        )


@dataclasses.dataclass(frozen=True)
class _MobileUNetConfig:

    model_factory: Callable[..., nn.Module]
    """Factory to create the base model."""

    weight_enum: WeightsEnum
    """Pre-trained weight enum."""

    channels: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    """A list of (in_channels, kernel_size, out_channels)."""

    downsample_layers: list[int] = dataclasses.field(default_factory=list)
    """Layer indices where spatial downsampling occurs."""

    expansion_layers: list[int] = dataclasses.field(default_factory=list)
    """Layer indices where channel expansion occurs."""

    encoder_keys: list[str] = dataclasses.field(default_factory=list)
    """Keys for the encoder feature maps."""


def _make_decoder_layers(
    cfg: _MobileUNetConfig,
    input_channels: int | None = None,
    upsample_channel_factor: int = 2,
):
    encoder = cfg.model_factory(weights=None).features
    decoder_layers = []

    for idx, layer in enumerate(encoder):  # type: ignore
        downsample = idx in cfg.downsample_layers
        expand = idx in cfg.expansion_layers
        new_layer = copy.deepcopy(layer)
        if downsample or expand:
            in_channels, kernel_size, out_channels = cfg.channels[idx]
            if idx == 0 and input_channels:
                in_channels = input_channels
            padding = (kernel_size - 1) // 2
            if isinstance(new_layer, misc.Conv2dNormActivation):
                if downsample:
                    if idx == 0:
                        new_layer = nn.ConvTranspose2d(
                            in_channels=upsample_channel_factor * out_channels,
                            out_channels=in_channels,
                            kernel_size=(kernel_size, kernel_size),
                            stride=(2, 2),
                            padding=(padding, padding),
                            output_padding=1,
                            bias=False,
                        )
                    else:
                        new_layer = _ConvTranspose2dNormActivation(
                            in_channels=upsample_channel_factor * out_channels,
                            out_channels=in_channels,
                            kernel_size=(kernel_size, kernel_size),
                            stride=(2, 2),
                            padding=(padding, padding),
                            bias=False,
                        )
                else:
                    new_layer = misc.Conv2dNormActivation(
                        in_channels=out_channels,
                        out_channels=in_channels,
                        kernel_size=(kernel_size, kernel_size),
                        padding=(padding, padding),
                        bias=False,
                    )
            elif isinstance(new_layer, mobilenetv3.InvertedResidual):
                next_channels = cfg.channels[idx + 1][0]
                if downsample:
                    next_channels *= upsample_channel_factor
                c0 = misc.Conv2dNormActivation(
                    in_channels=next_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    bias=False,
                )
                c3 = misc.Conv2dNormActivation(
                    in_channels=out_channels,
                    out_channels=in_channels,
                    kernel_size=(1, 1),
                    bias=False,
                )
                new_layer.block[0] = c0
                new_layer.block[-1] = c3

                if downsample:
                    new_layer.block[1] = _ConvTranspose2dNormActivation(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=(kernel_size, kernel_size),
                        stride=(2, 2),
                        padding=(padding, padding),
                        groups=out_channels,
                        bias=False,
                    )

            else:
                raise RuntimeError(f"Layer {idx}: Unexpected class: {new_layer.__class__.__name__}")

        decoder_layers.append(new_layer)

    return decoder_layers


class MobileV3UNet(nn.Module):

    def __init__(
        self,
        cfg: _MobileUNetConfig,
        in_channels: int = 3,
        out_channels: int = 1,
        *,
        pretrained: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        weights = cfg.weight_enum if pretrained else None
        encoder: nn.Sequential = cfg.model_factory(weights=weights).features  # type: ignore

        # replace only if input channels is not 3
        if in_channels != 3:
            with torch.no_grad():
                encoder[0][0] = nn.Conv2d(  # type: ignore
                    in_channels,
                    cfg.channels[0][-1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                )
        final_layer_idx = len(cfg.channels) - 1
        self.encoder = feature_extraction.create_feature_extractor(
            encoder,
            {str(idx): val for idx, val in zip(cfg.downsample_layers + [final_layer_idx], cfg.encoder_keys)},
        )

        decoder_layers = _make_decoder_layers(
            cfg=cfg,
            input_channels=out_channels,
            upsample_channel_factor=2,
        )
        sub_layers = [
            decoder_layers[i1 + 1 : i2 + 1] for i1, i2 in itertools.pairwise([-1] + cfg.downsample_layers + [final_layer_idx])
        ]
        self.decoder_blocks = nn.ModuleList([nn.Sequential(*reversed(layers)) for layers in sub_layers])
        self.num_blocks = len(self.decoder_blocks)

        self.dropout = nn.Dropout(dropout)
        self.cfg = cfg

        assert self.num_blocks == len(cfg.encoder_keys)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        out_dict = self.encoder(x)
        features = [out_dict[key] for key in self.cfg.encoder_keys]
        return features

    def decode(self, features: list[torch.Tensor]):
        z = self._transform(features[-1], self.num_blocks - 1)
        x_hat = self.decoder_blocks[-1](z)
        for block_idx in range(self.num_blocks - 2, -1, -1):
            z = self._transform(features[block_idx], block_idx)
            if tuple(z.shape[-2:]) != tuple(x_hat.shape[-2:]):
                x_hat = F.interpolate(x_hat, size=z.shape[-2:])
            block = self.decoder_blocks[block_idx]
            x_hat = torch.cat([x_hat, z], dim=1)
            x_hat = block(x_hat)

        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        x_hat = self.decode(features)
        return x_hat

    def _transform(self, x: torch.Tensor, idx: int):
        """Override this method for more sophisticated transforms."""
        x = self.dropout(x)
        return x


_MobileUNetLargeConfig = _MobileUNetConfig(
    model_factory=mobilenetv3.mobilenet_v3_large,
    weight_enum=mobilenetv3.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    channels=[
        (3, 3, 16),
        (16, 3, 16),
        (16, 3, 64),
        (24, 3, 72),
        (24, 5, 72),
        (40, 5, 120),
        (40, 5, 120),
        (40, 3, 240),
        (80, 3, 200),
        (80, 3, 184),
        (80, 3, 184),
        (80, 3, 480),
        (112, 3, 672),
        (112, 5, 672),
        (160, 5, 960),
        (160, 5, 960),
        (160, 1, 960),
    ],
    downsample_layers=[0, 2, 4, 7, 13],
    expansion_layers=[0, 2, 4, 7, 11, 13, 16],
    encoder_keys=[f"c{i + 1}" for i in range(6)],
)

_MobileUNetSmallConfig = _MobileUNetConfig(
    model_factory=mobilenetv3.mobilenet_v3_small,
    weight_enum=mobilenetv3.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    channels=[
        (3, 3, 16),
        (16, 3, 16),
        (16, 3, 72),
        (24, 3, 88),
        (24, 5, 96),
        (40, 5, 240),
        (40, 5, 240),
        (40, 5, 120),
        (48, 5, 144),
        (48, 5, 288),
        (96, 5, 576),
        (96, 5, 576),
        (96, 1, 576),
    ],
    downsample_layers=[0, 1, 2, 4, 9],
    expansion_layers=[0, 2, 4, 7, 9, 12],
    encoder_keys=[f"c{i + 1}" for i in range(6)],
)


def mobileunet_large(
    in_channels: int = 3,
    out_channels: int = 1,
    *,
    pretrained: bool = False,
    dropout: float = 0.0,
):
    return MobileV3UNet(
        cfg=_MobileUNetLargeConfig,
        in_channels=in_channels,
        out_channels=out_channels,
        pretrained=pretrained,
        dropout=dropout,
    )


def mobileunet_small(
    in_channels: int = 3,
    out_channels: int = 1,
    *,
    pretrained: bool = False,
    dropout: float = 0.0,
):
    return MobileV3UNet(
        cfg=_MobileUNetSmallConfig,
        in_channels=in_channels,
        out_channels=out_channels,
        pretrained=pretrained,
        dropout=dropout,
    )
