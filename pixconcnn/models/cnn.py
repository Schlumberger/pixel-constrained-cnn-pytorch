import torch.nn as nn
from pixconcnn.layers import ResidualBlock


class ResNet(nn.Module):
    """ResNet (with regular unmasked convolutions) mapping an conditional pixel
    inputs to logits.

    Parameters
    ----------
    img_size : tuple of ints
        Shape of input image. Note that since mask is appended to the masked
        image, the img_size given should be (num_channels + 1, height, width).
        Note, the extra channel used to store the mask.

    num_colors : int
        Number of colors to quantize output into. Typically 256, but can be
        lower for e.g. binary images.

    num_filters : int
        Number of filters for each convolution layer in model.

    depth : int
        Number of layers of model. Must be at least 2 to have an input and
        output layer.

    filter_size : int
        Size of convolutional filters.
    """
    def __init__(self, img_size=(2, 32, 32), num_colors=256, num_filters=32,
                 depth=17, filter_size=5):
        super(ResNet, self).__init__()

        self.depth = depth
        self.filter_size = filter_size
        self.padding = (filter_size - 1) // 2
        self.img_size = img_size
        self.num_channels = img_size[0] - 1  # Only output logits for color channels, not mask channel
        self.num_colors = num_colors
        self.num_filters = num_filters

        layers = [nn.Conv2d(self.num_channels + 1, self.num_filters,
                            self.filter_size, stride=1, padding=self.padding)]

        for _ in range(self.depth - 2):
            layers.append(
                ResidualBlock(self.num_filters, self.num_filters,
                              self.filter_size, stride=1, padding=self.padding)
            )

        # Final layer to output logits
        layers.append(
            nn.Conv2d(self.num_filters, self.num_colors * self.num_channels, 1)
        )

        self.img_to_pixel_logits = nn.Sequential(*layers)

    def forward(self, x):
        _, height, width = self.img_size
        # Shape (batch, output_channels, height, width)
        logits = self.img_to_pixel_logits(x)
        # Shape (batch, num_colors, channels, height, width)
        return logits.view(-1, self.num_colors, self.num_channels, height, width)
