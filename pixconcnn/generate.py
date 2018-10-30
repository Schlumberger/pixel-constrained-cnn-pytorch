import numpy as np
from torch import device as torch_device
from torch import zeros as torch_zeros
from torch.cuda import is_available as cuda_is_available
from torchvision.utils import make_grid
from utils.masks import MaskGenerator, get_repeated_conditional_pixels


def generate_images(model, batch, mask_descriptors, num_samples=64, temp=1.,
                    verbose=False):
    """Generates image completions based on the images in batch masked by the
    masks in mask_descriptors. This will generate
    batch.size(0) * len(mask_descriptors) * num_samples completions, i.e.
    num_samples completions for every image and mask combination.

    Parameters
    ----------
    model : pixconcnn.models.pixel_constrained.PixelConstrained instance

    batch : torch.Tensor

    mask_descriptors : list of mask_descriptor
        See utils.masks.MaskGenerator for allowed mask_descriptors.

    num_samples : int
        Number of samples to generate for a given image-mask combination.

    temp : float
        Temperature for sampling.

    verbose : bool
        If True prints progress information while generating images
    """
    device = torch_device("cuda" if cuda_is_available() else "cpu")
    model.to(device)
    outputs = []
    for i in range(batch.size(0)):
        outputs_per_img = []
        for j in range(len(mask_descriptors)):
            if verbose:
                print("Generating samples for image {} using mask {}".format(i, mask_descriptors[j]))
            # Get image and mask combination
            img = batch[i:i+1]
            mask_generator = MaskGenerator(model.prior_net.img_size, mask_descriptors[j])
            mask = mask_generator.get_masks(1)
            # Create conditional pixels which will be used to sample completions
            cond_pixels = get_repeated_conditional_pixels(img, mask, model.prior_net.num_colors, num_samples)
            cond_pixels = cond_pixels.to(device)
            samples, log_probs = model.sample(cond_pixels, return_likelihood=True, temp=temp)
            outputs_per_img.append({
                "orig_img": img,
                "cond_pixels": cond_pixels,
                "mask": mask,
                "samples": samples,
                "log_probs": log_probs
            })
        outputs.append(outputs_per_img)
    return outputs
