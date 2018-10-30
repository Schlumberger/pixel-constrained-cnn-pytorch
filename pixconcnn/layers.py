import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block (note that number of in_channels and out_channels must be
    the same).

    Parameters
    ----------
    in_channels : int

    out_channels : int

    kernel_size : int or tuple of ints

    stride : int

    padding : int
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        # In and out channels should be the same
        return x + self.convs(x)


class MaskedConv2d(nn.Conv2d):
    """
    Implements various 2d masked convolutions.

    Parameters
    ----------
    mask_type : string
        Defines the type of mask to use. One of 'A', 'A_Red', 'A_Green',
        'A_Blue', 'B', 'B_Red', 'B_Green', 'B_Blue', 'H', 'H_Red', 'H_Green',
        'H_Blue', 'HR', 'HR_Red', 'HR_Green', 'HR_Blue', 'V', 'VR'.
    """
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.mask_type = mask_type

        # Initialize mask
        mask = torch.zeros(*self.weight.size())
        _, kernel_c, kernel_h, kernel_w = self.weight.size()
        # If using a color mask, the number of channels must be divisible by 3
        if mask_type.endswith('Red') or mask_type.endswith('Green') or mask_type.endswith('Blue'):
            assert kernel_c % 3 == 0
        # If using a horizontal mask, the kernel height must be 1
        if mask_type.startswith('H'):
            assert kernel_h == 1  # Kernel should have shape (1, kernel_w)

        if mask_type == 'A':
            # For 3 by 3 kernel, this would be:
            #  1 1 1
            #  1 0 0
            #  0 0 0
            mask[:, :, kernel_h // 2, :kernel_w // 2] = 1.
            mask[:, :, :kernel_h // 2, :] = 1.
        elif mask_type == 'A_Red':
            # Mask type A for red channels. Same as regular mask A
            if kernel_h == 1 and kernel_w == 1:
                pass  # Mask is all zeros for 1x1 convolution
            else:
                mask[:, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :kernel_h // 2, :] = 1.
        elif mask_type == 'A_Green':
            # Mask type A for green channels. Same as regular mask A, except
            # central pixel of first third of channels is 1
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :kernel_c // 3, 0, 0] = 1.
            else:
                mask[:, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :kernel_h // 2, :] = 1.
                mask[:, :kernel_c // 3, kernel_h // 2, kernel_w // 2] = 1.
        elif mask_type == 'A_Blue':
            # Mask type A for blue channels. Same as regular mask A, except
            # central pixel of first two thirds of channels is 1
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :2 * kernel_c // 3, 0, 0] = 1.
            else:
                mask[:, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :kernel_h // 2, :] = 1.
                mask[:, :2 * kernel_c // 3, kernel_h // 2, kernel_w // 2] = 1.
        elif mask_type == 'B':
            # For 3 by 3 kernel, this would be:
            #  1 1 1
            #  1 1 0
            #  0 0 0
            mask[:, :, kernel_h // 2, :kernel_w // 2 + 1] = 1.
            mask[:, :, :kernel_h // 2, :] = 1.
        elif mask_type == 'B_Red':
            # Mask type B for red channels. Same as regular mask B, except last
            # two thirds of channels of central pixels are 0. Alternatively,
            # same as Mask A but with first third of channels of central pixels
            # are 1
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :kernel_c // 3, 0, 0] = 1.
            else:
                mask[:, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :kernel_h // 2, :] = 1.
                mask[:, :kernel_c // 3, kernel_h // 2, kernel_w // 2] = 1.
        elif mask_type == 'B_Green':
            # Mask type B for green channels. Same as regular mask B, except
            # last third of channels of central pixels are 0
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :2 * kernel_c // 3, 0, 0] = 1.
            else:
                mask[:, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :kernel_h // 2, :] = 1.
                mask[:, :2 * kernel_c // 3, kernel_h // 2, kernel_w // 2] = 1.
        elif mask_type == 'B_Blue':
            # Mask type B for blue channels. Same as regular mask B
            if kernel_h == 1 and kernel_w == 1:
                mask[:, :, 0, 0] = 1.
            else:
                mask[:, :, kernel_h // 2, :kernel_w // 2] = 1.
                mask[:, :, :kernel_h // 2, :] = 1.
                mask[:, :, kernel_h // 2, kernel_w // 2] = 1.
        elif mask_type == 'H':
            # For 3 by 3 kernel, this would be:
            #  1 1 0
            # Mask for horizontal stack in regular gated conv
            mask[:, :, 0, :kernel_w // 2 + 1] = 1.
        elif mask_type == 'H_Red':
            mask[:, :, 0, :kernel_w // 2] = 1.
            mask[:, :kernel_c // 3, 0, kernel_w // 2] = 1.
        elif mask_type == 'H_Green':
            mask[:, :, 0, :kernel_w // 2] = 1.
            mask[:, :2 * kernel_c // 3, 0, kernel_w // 2] = 1.
        elif mask_type == 'H_Blue':
            mask[:, :, 0, :kernel_w // 2] = 1.
            mask[:, :, 0, kernel_w // 2] = 1.
        elif mask_type == 'HR':
            # For 3 by 3 kernel, this would be:
            #  1 0 0
            # Mask for horizontal stack in restricted gated conv
            mask[:, :, 0, :kernel_w // 2] = 1.
        elif mask_type == 'HR_Red':
            mask[:, :, 0, :kernel_w // 2] = 1.
        elif mask_type == 'HR_Green':
            mask[:, :, 0, :kernel_w // 2] = 1.
            mask[:, :kernel_c // 3, 0, kernel_w // 2] = 1.
        elif mask_type == 'HR_Blue':
            mask[:, :, 0, :kernel_w // 2] = 1.
            mask[:, :2 * kernel_c // 3, 0, kernel_w // 2] = 1.
        elif mask_type == 'V':
            # For 3 by 3 kernel, this would be:
            #  1 1 1
            #  1 1 1
            #  0 0 0
            mask[:, :, :kernel_h // 2 + 1, :] = 1.
        elif mask_type == 'VR':
            # For 3 by 3 kernel, this would be:
            #  1 1 1
            #  0 0 0
            #  0 0 0
            mask[:, :, :kernel_h // 2, :] = 1.

        # Register buffer adds a key to the state dict of the model. This will
        # track the attribute without registering it as a learnable parameter.
        # We require this since mask will be used in the forward pass.
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class MaskedConvRGB(nn.Module):
    """
    Masked convolution with RGB channel splitting.

    Parameters
    ----------
    mask_type : string
        One of 'A', 'B', 'V' or 'H'.

    in_channels : int
        Must be divisible by 3

    out_channels : int
        Must be divisible by 3

    kernel_size : int or tuple of ints

    stride : int

    padding : int

    bias : bool
        If True adds a bias term to the convolution.
    """
    def __init__(self, mask_type, in_channels, out_channels, kernel_size,
                 stride, padding, bias):
        super(MaskedConvRGB, self).__init__()

        self.conv_R = MaskedConv2d(mask_type + '_Red', in_channels,
                                   out_channels // 3, kernel_size,
                                   stride=stride, padding=padding, bias=bias)
        self.conv_G = MaskedConv2d(mask_type + '_Green', in_channels,
                                   out_channels // 3, kernel_size,
                                   stride=stride, padding=padding, bias=bias)
        self.conv_B = MaskedConv2d(mask_type + '_Blue', in_channels,
                                   out_channels // 3, kernel_size,
                                   stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out_red = self.conv_R(x)
        out_green = self.conv_G(x)
        out_blue = self.conv_B(x)
        return torch.cat([out_red, out_green, out_blue], dim=1)


class GatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 restricted=False):
        """Gated PixelCNN convolutional block. Note the number of input and
        output channels must be the same, unless restricted is True.

        Parameters
        ----------
        in_channels : int

        out_channels : int

        kernel_size : int
            Note this MUST be int and not tuple like in regular convs.

        stride : int

        padding : int

        restricted : bool
            If True, uses restricted masks, otherwise uses regular masks.
        """
        super(GatedConvBlock, self).__init__()

        assert type(kernel_size) is int

        self.restricted = restricted

        if restricted:
            vertical_mask = 'VR'
            horizontal_mask = 'HR'
        else:
            vertical_mask = 'V'
            horizontal_mask = 'H'

        self.vertical_conv = MaskedConv2d(vertical_mask, in_channels,
                                          2 * out_channels, kernel_size,
                                          stride=stride, padding=padding)

        self.horizontal_conv = MaskedConv2d(horizontal_mask, in_channels,
                                            2 * out_channels, (1, kernel_size),
                                            stride=stride, padding=(0, padding))

        self.vertical_to_horizontal = nn.Conv2d(2 * out_channels,
                                                2 * out_channels, 1)

        self.horizontal_conv_2 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, v_input, h_input):
        # Vertical stack
        v_conv = self.vertical_conv(v_input)
        v_out = gated_activation(v_conv)
        # Vertical to horizontal
        v_to_h = self.vertical_to_horizontal(v_conv)
        # Horizontal stack
        h_conv = self.horizontal_conv(h_input)
        h_conv_activation = gated_activation(h_conv + v_to_h)
        h_conv2 = self.horizontal_conv_2(h_conv_activation)
        if self.restricted:
            h_out = h_conv2
        else:
            h_out = h_conv2 + h_input
        return v_out, h_out


class GatedConvBlockRGB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 restricted=False):
        """Gated PixelCNN convolutional block for RGB images. Note the number of
        input and output channels must be the same, unless restricted is True.

        Parameters
        ----------
        in_channels : int

        out_channels : int

        kernel_size : int
            Note this MUST be int and not tuple like in regular convs.

        stride : int

        padding : int

        restricted : bool
            If True, uses restricted masks, otherwise uses regular masks.
        """
        super(GatedConvBlockRGB, self).__init__()

        assert type(kernel_size) is int

        self.restricted = restricted
        self.out_channels = out_channels

        if restricted:
            vertical_mask = 'VR'
            horizontal_mask = 'HR'
        else:
            vertical_mask = 'V'
            horizontal_mask = 'H'

        self.vertical_conv = MaskedConv2d(vertical_mask, in_channels,
                                          2 * out_channels, kernel_size,
                                          stride=stride, padding=padding)

        self.horizontal_conv = MaskedConvRGB(horizontal_mask, in_channels,
                                             2 * out_channels, (1, kernel_size),
                                             stride=stride,
                                             padding=(0, padding), bias=True)

        self.vertical_to_horizontal = nn.Conv2d(2 * out_channels,
                                                2 * out_channels, 1)

        self.horizontal_conv_2 = MaskedConvRGB('B', out_channels, out_channels,
                                               (1, 1), stride=1, padding=0,
                                               bias=True)

    def forward(self, v_input, h_input):
        # Vertical stack
        v_conv = self.vertical_conv(v_input)
        v_out = gated_activation(v_conv)
        # Vertical to horizontal
        v_to_h = self.vertical_to_horizontal(v_conv)
        # Horizontal stack
        h_conv = self.horizontal_conv(h_input) + v_to_h
        # Gated activation must be applied for the R, G and B part of the
        # convolutional volume separately to avoid information from different
        # channels leaking
        channels_third = 2 * self.out_channels // 3
        h_conv_activation_R = gated_activation(h_conv[:, :channels_third])
        h_conv_activation_G = gated_activation(h_conv[:, channels_third:2 * channels_third])
        h_conv_activation_B = gated_activation(h_conv[:, 2 * channels_third:])
        h_conv_activation = torch.cat([h_conv_activation_R,
                                       h_conv_activation_G,
                                       h_conv_activation_B],
                                       dim=1)
        # 1 by 1 convolution on horizontal stack
        h_conv2 = self.horizontal_conv_2(h_conv_activation)
        if self.restricted:
            h_out = h_conv2
        else:
            h_out = h_conv2 + h_input
        return v_out, h_out


def gated_activation(input_vol):
    """Applies a gated activation to the convolutional volume. Note that this
    activation divides the number of channels by 2.

    Parameters
    ----------
    input_vol : torch.Tensor
        Input convolutional volume. Shape (batch_size, channels, height, width)
        Note that number of channels must be even.

    Returns
    -------
    output_vol of shape (batch_size, channels // 2, height, width)
    """
    # Extract number of channels from input volume
    channels = input_vol.size(1)
    # Get activations for first and second half of volume
    tanh_activation = torch.tanh(input_vol[:, channels // 2:])
    sigmoid_activation = torch.sigmoid(input_vol[:, :channels // 2])
    return tanh_activation * sigmoid_activation
