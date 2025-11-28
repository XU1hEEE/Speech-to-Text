import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class WhiteNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class MeanDriftNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        _, C = x.shape
        noise = torch.randn(1, C) * self.std
        return x + noise

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")


class TimeMask(nn.Module):
    """
    SpecAugment-style time masking for neural time series.
    Randomly masks out consecutive time steps.

    Expected input shape: (C, T)
    - C = channels/electrodes
    - T = time steps
    """
    def __init__(self, max_mask_len=25, n_masks=2, mask_value=0.0):
        super().__init__()
        self.max_mask_len = max_mask_len
        self.n_masks = n_masks
        self.mask_value = mask_value

    def forward(self, x):
        """
        Args:
            x: Tensor (C, T)
        Returns:
            Masked tensor of same shape
        """
        if not self.training:
            return x

        C, T = x.shape
        x = x.clone()   # avoid in-place ops; safer for CTC & autograd

        for _ in range(self.n_masks):
            # random mask length
            mask_len = torch.randint(
                1, self.max_mask_len + 1, 
                (1,), device=x.device
            ).item()

            # random start index
            max_start = max(1, T - mask_len)
            start = torch.randint(
                0, max_start, 
                (1,), device=x.device
            ).item()

            # apply mask
            x[:, start:start + mask_len] = self.mask_value

        return x


class ChannelDropout(nn.Module):
    """
    Randomly drops entire channels (electrodes).
    Good for robustness to impedance spikes or bad contacts.
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        C = x.shape[0]
        drop_mask = (torch.rand(C) > self.p).float().to(x.device)
        drop_mask = drop_mask.unsqueeze(-1)
        return x * drop_mask


# class Compose(nn.Module):
#     """
#     Sequentially apply a list of augmentations.
#     """
#     def __init__(self, transforms):
#         super().__init__()
#         self.transforms = nn.ModuleList(transforms)

#     def forward(self, x):
#         for t in self.transforms:
#             x = t(x)
#         return x
