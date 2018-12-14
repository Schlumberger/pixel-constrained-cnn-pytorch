from matplotlib.pyplot import get_cmap
import numpy as np
import torch


def probs_and_conditional_plot(img, probs, mask, cmap='plasma'):
    """Creates a plot of pixel probabilities with the conditional pixels from
    the original image overlayed. Note this function only works for binary
    images.

    Parameters
    ----------
    img : torch.Tensor
        Shape (1, H, W)

    probs : torch.Tensor
        Shape (H, W). Should be the probability of a pixel being 1.

    mask : torch.Tensor
        Shape (H, W)

    cmap : string
        Colormap to use for probs (as defined in matplotlib.plt,
        e.g. 'jet', 'viridis', ...)
    """
    # Define function to convert array to colormap rgb image
    # The colorscale has a min value of 0 and a max value of 1 by default
    # (i.e. it does not rescale depending on the value of the probs)
    convert_to_cmap = get_cmap(cmap)
    # Create output image from colormap of probs
    rgba_probs = convert_to_cmap(probs.numpy())
    output_img = np.delete(rgba_probs, 3, 2) # Convert to RGB
    # Overlay unmasked parts of original image over probs
    np_mask = mask.numpy().astype(bool)  # Convert mask to boolean numpy array
    np_img = img.numpy()[0]  # Convert img to grayscale numpy img
    output_img[:, :, 0][np_mask] = np_img[np_mask]
    output_img[:, :, 1][np_mask] = np_img[np_mask]
    output_img[:, :, 2][np_mask] = np_img[np_mask]
    # Convert numpy image to torch tensor
    return torch.Tensor(output_img.transpose(2, 0, 1))


def uncertainty_plot(samples, log_probs, cmap='plasma'):
    """Sorts samples by their log likelihoods and creates an image representing
    the log likelihood of each sample as a box with color and size proportional
    to the log likelihood.

    Parameters
    ----------
    samples : torch.Tensor
        Shape (N, C, H, W)

    log_probs : torch.Tensor
        Shape (N,)

    cmap : string
        Colormap to use for likelihoods (as defined in matplotlib.plt,
        e.g. 'jet', 'viridis', ...)
    """
    # Sorted by negative log likelihood
    sorted_nll, sorted_indices = torch.sort(-log_probs)
    sorted_samples = samples[sorted_indices]
    # Normalize log likelihoods to be in 0 - 1 range
    min_ll, max_ll = (-sorted_nll).min(), (-sorted_nll).max()
    normalized_likelihoods = ((-sorted_nll) - min_ll) / (max_ll - min_ll)

    # For each sample draw an image with a box proportional in size and
    # color to the log likelihood value
    num_samples, _, height, width = samples.size()
    # Initialize white background images on which to draw boxes
    ll_images = torch.ones(num_samples, 3, height, width)
    # Specify box sizes
    lower_width = width // 2 - width // 5
    upper_width = width // 2 + width // 5
    max_box_height = height
    min_box_height = 1
    # Generate colors for the boxes
    convert_to_cmap = get_cmap(cmap)
    # Remove alpha channel from colormap
    colors = convert_to_cmap(normalized_likelihoods.numpy())[:, :-1]

    # Fill out images with boxes
    for i in range(num_samples):
        norm_ll = normalized_likelihoods[i].item()
        box_height = int(min_box_height + (max_box_height - min_box_height) * norm_ll)
        box_color = colors[i]
        for j in range(3):
            ll_images[i, j, height - box_height:height, lower_width:upper_width] = box_color[j]

    return sorted_samples, ll_images
