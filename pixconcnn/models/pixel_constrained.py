import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelConstrained(nn.Module):
    """Pixel Constrained CNN model.

    Parameters
    ----------
    prior_net : pixconcnn.models.gated_pixelcnn.GatedPixelCNN(RGB) instance
        Model defining the prior network.

    cond_net : pixconcnn.models.cnn.ResNet instance
        Model defining the conditioning network.
    """
    def __init__(self, prior_net, cond_net):

        super(PixelConstrained, self).__init__()

        self.prior_net = prior_net
        self.cond_net = cond_net

    def forward(self, x, x_cond):
        """
        x : torch.Tensor
            Image to predict logits for.

        x_cond : torch.Tensor
            Image containing pixels to be conditioned on. The pixels which are
            being conditioned on should retain their usual value, while all
            other pixels should be set to 0. In addition the mask should be
            appended to known pixels such that the shape of x_cond is
            (batch_size, channels + 1, height, width). For more details, see
            utils.masks.get_conditional_pixels function.
        """
        prior_logits = self.prior_net(x)
        cond_logits = self.cond_net(x_cond)
        logits = prior_logits + cond_logits
        return logits, prior_logits, cond_logits

    def sample(self, x_cond, temp=1., return_likelihood=False):
        """Generate conditional samples from the model. The number of samples
        generated will be equal to the batch size of x_cond.

        Parameters
        ----------
        x_cond : torch.Tensor
            Tensor containing the conditioning pixels. Should have shape
            (num_samples, channels + 1, height, width).

        temp : float
            Temperature of softmax distribution. Temperatures larger than 1
            make the distribution more uniforman while temperatures lower than
            1 make the distribution more peaky.

        return_likelihood : bool
            If True returns the log likelihood of the samples according to the
            model.
        """
        # Set model to evaluation mode
        self.eval()

        # "Channel dimension" of x_cond has size channels + 1, so decrease this
        num_samples, channels_plus_mask, height, width = x_cond.size()
        channels = channels_plus_mask - 1

        # Samples to be generated
        samples = torch.zeros((num_samples, channels, height, width))
        # Move samples to same device as conditional tensor
        samples = samples.to(x_cond.device)
        num_colors = self.prior_net.num_colors

        if return_likelihood:
            log_probs = torch.zeros(num_samples)
            log_probs = log_probs.to(x_cond.device)

        # Sample pixel intensities from a batch of probability distributions
        # for each pixel in each channel
        with torch.no_grad():
            for i in range(height):
                for j in range(width):
                    # The unmasked pixels are the ones where the mask has
                    # nonzero value
                    unmasked = x_cond[:, -1, i, j] > 0
                    # If (i, j)th pixel is known for all images in batch (i.e.
                    # if all values in unmasked are True), do not perform
                    # forward pass of the model
                    sample_pixel = True
                    if unmasked.long().sum().item() == num_samples:
                        sample_pixel = False
                    for k in range(channels):
                        if sample_pixel:
                            logits, _, _ = self.forward(samples, x_cond)
                            probs = F.softmax(logits / temp, dim=1)
                            # Note that probs has shape
                            # (batch, num_colors, channels, height, width)
                            pixel_val = torch.multinomial(probs[:, :, k, i, j], 1)
                            # The pixel intensities will be given by 0, 1, 2, ..., so
                            # normalize these to be in 0 - 1 range as this is what the
                            # model expects
                            samples[:, k, i, j] = pixel_val[:, 0].float() / (num_colors - 1)
                        # Set all the unmasked pixels to the value they are
                        # conditioned on
                        samples[:, k, i, j][unmasked] = x_cond[:, k, i, j][unmasked]

                        # Add log prob of current pixel
                        if return_likelihood:
                            if sample_pixel:
                                # Probabilities for the given pixel
                                probs_pixel = probs[:, pixel_val[:, 0], k, i, j][:, 0]
                                # Set probabilities of unmasked pixels to 1 since
                                # they are already known
                                probs_pixel[unmasked] = 1.
                            else:
                                # If sample_pixel is False, all pixels in batch
                                # are known, so all probabilities are 1
                                probs_pixel = torch.ones(num_samples)
                                probs_pixel = probs_pixel.to(x_cond.device)
                            # Add log probs (1e-9 to avoid log(0))
                            log_probs += torch.log(probs_pixel + 1e-9)

        # Reset model to train mode
        self.train()

        # Unnormalize pixels
        samples = (samples * (num_colors - 1)).long()

        if return_likelihood:
            return samples.cpu(), log_probs.cpu()
        else:
            return samples.cpu()

    def sample_unconditional(self, device, num_samples=16):
        """Samples from prior model without conditioning."""
        return self.prior_net.sample(device, num_samples)
