import typing
from torch import nn
from abc import abstractmethod
import torch


class BaseEncoder(nn.Module):
    def __init__(self) -> None:
        super(BaseEncoder, self).__init__()

    def forward(self, x: torch.tensor) -> torch.tensor:
        raise NotImplementedError


class BaseDecoder(nn.Module):
    def __init__(self) -> None:
        super(BaseDecoder, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BaseVarAutoencoder(nn.Module):
    def __init__(self) -> None:
        super(BaseVarAutoencoder, self).__init__()

    def sample(self, batch_size: int, current_device: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> [torch.Tensor, ...]:
        pass

    @abstractmethod
    def loss(self, *inputs: typing.Any, **kwargs) -> dict:
        pass


class BaseConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple[int, ...],
                 stride: tuple[int, ...],
                 padding: int,
                 max_pool: bool,
                 **kwargs):

        super(BaseConvBlock, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

        if max_pool:
            self.max_pooling = nn.MaxPool2d(kernel_size=kwargs["kernel_max_pool"], stride=kwargs["stride_max_pool"])
        else:
            # if we dont want to use maxpool, set kernel and stride to 1
            self.max_pooling = nn.MaxPool2d(kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> [torch.Tensor, ...]:
        out = self.conv_layer(x)
        out = self.max_pooling(out)
        out = self.batch_norm(out)
        return self.relu(out)


class BaseTransposeConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple[int, ...],
                 stride: tuple[int, ...],
                 padding: int,
                 upsample: bool,
                 **kwargs):

        super(BaseTransposeConvBlock, self).__init__()
        self.transpose_conv_layer = nn.ConvTranspose2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding,
                                                       output_padding=1)

        if upsample:
            self.upsample = nn.Upsample(scale_factor=kwargs["scale_factor"])
        else:
            # if we dont want to use upsample, set kernel and stride to 1
            self.upsample = nn.Upsample(scale_factor=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> [torch.Tensor, ...]:
        out = self.transpose_conv_layer(x)
        out = self.upsample(out)
        out = self.batch_norm(out)
        return self.relu(out)



