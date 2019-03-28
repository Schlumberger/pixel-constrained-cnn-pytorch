import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from pixconcnn.layers import GatedConvBlock, GatedConvBlockRGB, MaskedConvRGB


class PixelCNNBaseClass(nn.Module):
    """Abstract class defining PixelCNN sampling which is the same for both
    single channel and RGB models.
    """
    __metaclass__ = ABCMeta
    @abstractmethod
    def forward(self):
        """Forward method is implemented in child class (GatedPixelCNN or
        GatedPixelCNNRGB)."""
        pass

    def sample(self, device, num_samples=16, temp=1., return_likelihood=False):
        """Generates samples from a GatedPixelCNN or GatedPixelCNNRGB.

        Parameters
        ----------
        device : torch.device instance

        num_samples : int
            Number of samples to generate

        temp : float
            Temperature of softmax distribution. Temperatures larger than 1
            make the distribution more uniform while temperatures lower than
            1 make the distribution more peaky.

        return_likelihood : bool
            If True returns the log likelihood of the samples according to the
            model.
        """
        # Set model to evaluation mode
        self.eval()

        samples = torch.zeros(num_samples, *self.img_size)
        samples = samples.to(device)
        channels, height, width = self.img_size

        # Sample pixel intensities from a batch of probability distributions
        # for each pixel in each channel
        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    for k in range(channels):
                        logits = self.forward(samples)
                        probs = F.softmax(logits / temp, dim=1)
                        # Note that probs has shape
                        # (batch, num_colors, channels, height, width)
                        pixel_val = torch.multinomial(probs[:, :, k, i, j], 1)
                        # The pixel intensities will be given by 0, 1, 2, ..., so
                        # normalize these to be in 0 - 1 range as this is what the
                        # model expects. Note that pixel_val has shape (batch, 1)
                        # so remove last dimension
                        samples[:, k, i, j] = pixel_val[:, 0].float() / (self.num_colors - 1)

        # Reset model to train mode
        self.train()

        # Unnormalize pixels
        samples = (samples * (self.num_colors - 1)).long()

        if return_likelihood:
            return samples.cpu(), self.log_likelihood(device, samples).cpu()
        else:
            return samples.cpu()

    def log_likelihood(self, device, samples):
        """Calculates log likelihood of samples under model.

        Parameters
        ----------
        device : torch.device instance

        samples : torch.Tensor
            Batch of images. Shape (batch_size, num_channels, width, height).
            Values should be integers in [0, self.prior_net.num_colors - 1].
        """
        # Set model to evaluation mode
        self.eval()

        num_samples, num_channels, height, width = samples.size()
        log_probs = torch.zeros(num_samples)
        log_probs = log_probs.to(device)

        # Normalize samples before passing through model
        norm_samples = samples.float() / (self.num_colors - 1)
        # Calculate pixel probs according to the model
        logits = self.forward(norm_samples)
        # Note that probs has shape
        # (batch, num_colors, channels, height, width)
        probs = F.softmax(logits, dim=1)

        # Calculate probability of each pixel
        for i in range(height):
            for j in range(width):
                for k in range(num_channels):
                    # Get the batch of true values at pixel (k, i, j)
                    true_vals = samples[:, k, i, j]
                    # Get probability assigned by model to true pixel
                    probs_pixel = probs[:, true_vals, k, i, j][:, 0]
                    # Add log probs (1e-9 to avoid log(0))
                    log_probs += torch.log(probs_pixel + 1e-9)

        # Reset model to train mode
        self.train()

        return log_probs



class GatedPixelCNN(PixelCNNBaseClass):
    """Gated PixelCNN model for single channel images.

    Parameters
    ----------
    img_size : tuple of ints
        Shape of input image. E.g. (1, 32, 32)

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
    def __init__(self, img_size=(1, 32, 32), num_colors=256, num_filters=64,
                 depth=17, filter_size=5):
        super(GatedPixelCNN, self).__init__()

        self.depth = depth
        self.filter_size = filter_size
        self.padding = (filter_size - 1) // 2
        self.img_size = img_size
        self.num_channels = img_size[0]
        self.num_colors = num_colors
        self.num_filters = num_filters

        # First layer is restricted (to avoid self-dependency on input pixel)
        self.input_to_stacks = GatedConvBlock(self.num_channels,
                                              self.num_filters,
                                              self.filter_size,
                                              stride=1,
                                              padding=self.padding,
                                              restricted=True)

        # Subsequent layers are regular gated blocks for vertical and horizontal
        # stack
        gated_stacks = []
        # -2 since we are not counting first and last layer
        for _ in range(self.depth - 2):
            gated_stacks.append(
                GatedConvBlock(self.num_filters, self.num_filters,
                               self.filter_size, stride=1, padding=self.padding)
            )
        self.gated_stacks = nn.ModuleList(gated_stacks)

        # Final layer to output logits
        self.stacks_to_pixel_logits = nn.Conv2d(self.num_filters, self.num_colors * self.num_channels, 1)

    def forward(self, x):
        # Restricted gated layer
        vertical, horizontal = self.input_to_stacks(x, x)
        # Iterate over gated layers
        for gated_block in self.gated_stacks:
            vertical, horizontal = gated_block(vertical, horizontal)
        # Output logits from the horizontal stack (i.e. the stack which receives
        # information from both horizontal and vertical stack)
        logits = self.stacks_to_pixel_logits(horizontal)

        # Reshape logits
        _, height, width = self.img_size
        # Shape (batch, output_channels, height, width) ->
        # (batch, num_colors, channels, height, width)
        return logits.view(-1, self.num_colors, self.num_channels, height, width)


class GatedPixelCNNRGB(PixelCNNBaseClass):
    """Gated PixelCNN model for RGB images.

    Parameters
    ----------
    img_size : tuple of ints
        Shape of input image. E.g. (3, 32, 32)

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
    def __init__(self, img_size=(3, 32, 32), num_colors=256, num_filters=64,
                 depth=17, filter_size=5):
        super(GatedPixelCNNRGB, self).__init__()

        self.depth = depth
        self.filter_size = filter_size
        self.padding = (filter_size - 1) / 2
        self.img_size = img_size
        self.num_channels = img_size[0]
        self.num_colors = num_colors
        self.num_filters = num_filters

        # First layer is restricted (to avoid self-dependency on input pixel)
        self.input_to_stacks = GatedConvBlockRGB(self.num_channels,
                                                 self.num_filters,
                                                 self.filter_size,
                                                 stride=1,
                                                 padding=self.padding,
                                                 restricted=True)

        # Subsequent layers are regular gated blocks for vertical and horizontal
        # stack
        gated_stacks = []
        # -2 since we are not counting first and last layer
        for _ in range(self.depth - 2):
            gated_stacks.append(
                GatedConvBlockRGB(self.num_filters, self.num_filters,
                                  self.filter_size, stride=1,
                                  padding=self.padding)
            )
        self.gated_stacks = nn.ModuleList(gated_stacks)

        # Final layer to output logits
        self.stacks_to_pixel_logits = nn.Sequential(
            MaskedConvRGB('B', self.num_filters, 1023, (1, 1), stride=1, padding=0, bias=True),
            nn.ReLU(True),
            MaskedConvRGB('B', 1023, self.num_colors * self.num_channels, (1, 1), stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        # Restricted gated layer
        vertical, horizontal = self.input_to_stacks(x, x)
        # Several gated layers
        for gated_block in self.gated_stacks:
            vertical, horizontal = gated_block(vertical, horizontal)
        # Output logits from the horizontal stack (i.e. the stack which receives
        # information from both horizontal and vertical stack)
        logits = self.stacks_to_pixel_logits(horizontal)

        # Reshape logits maintaining order between R, G and B channels
        _, height, width = self.img_size
        logits_red = logits[:, :self.num_colors]
        logits_green = logits[:, self.num_colors:2 * self.num_colors]
        logits_blue = logits[:, 2 * self.num_colors:]
        logits_red = logits_red.view(-1, self.num_colors, 1, height, width)
        logits_green = logits_green.view(-1, self.num_colors, 1, height, width)
        logits_blue = logits_blue.view(-1, self.num_colors, 1, height, width)
        # Shape (batch, num_colors, channels, height, width)
        return torch.cat([logits_red, logits_green, logits_blue], dim=2)
