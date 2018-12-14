import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class MaskGenerator():
    """Class used to generate masks. Can be used to create masks during
    training or to build various masks for generation.

    Parameters
    ----------
    img_size : tuple of ints
        E.g. (1, 28, 28) or (3, 64, 64)

    mask_descriptor : tuple of string and other
        Mask descriptors will be of the form (mask_type, mask_attribute).
        Allowed descriptors are:
        1. ('random', None or int or tuple of ints): Generates random masks,
            where the position of visible pixels is selected uniformly
            at random over the image. If mask_attribute is None then the
            number of visible pixels is sampled uniformly between 1 and the
            total number of pixels in the image, otherwise it is fixed to
            the int given in mask_attribute. If mask_attribute is a tuple
            of ints, the number of visible pixels is sampled uniformly
            between the first int (lower bound) and the second int (upper
            bound).
        2. ('bottom', int): Generates masks where only the bottom pixels are
            visible. The int determines the number of rows of the image to
            keep visible at the bottom.
        3. ('top', int): Generates masks where only the top pixels are
            visible. The int determines the number of rows of the image to
            keep visible at the top.
        4. ('center', int): Generates masks where only the central pixels
            are visible. The int determines the size in pixels of the sides
            of the square of visible pixels of the image.
        5. ('edge', int): Generates masks where only the edge pixels of the
            image are visible. The int determines the thickness of the edges
            in pixels.
        6. ('left', int): Generates masks where only the left pixels of the
            image are visible. The int determines the number of columns
            in pixels which are visible.
        7. ('right', int): Generates masks where only the right pixels of
            the image are visible. The int determines the number of columns
            in pixels which are visible.
        8. ('random_rect', (int, int)): Generates random rectangular masks
            where the maximum height and width of the rectangles are
            determined by the two ints.
        9. ('random_blob', (int, (int, int), float)): Generates random
            blobs, where the number of blobs is determined by the first int,
            the range of iterations (see function definition) is determined
            by the tuple of ints and the threshold for making pixels visible
            is determined by the float.
        10. ('random_blob_cache', (str, int)): Loads pregenerated random masks
            from a folder given by the string, using a batch_size given by
            the int.
    """
    def __init__(self, img_size, mask_descriptor):
        self.img_size = img_size
        self.num_pixels = img_size[1] * img_size[2]
        self.mask_type, self.mask_attribute = mask_descriptor

        if self.mask_type == 'random_blob_cache':
            dset = datasets.ImageFolder(self.mask_attribute[0],
                                        transform=transforms.Compose([transforms.Grayscale(),
                                                                      transforms.ToTensor()]))
            self.data_loader = DataLoader(dset, batch_size=self.mask_attribute[1], shuffle=True)

    def get_masks(self, batch_size):
        """Returns a tensor of shape (batch_size, 1, img_size[1], img_size[2])
        containing masks which were generated according to mask_type and
        mask_attribute.

        Parameters
        ----------
        batch_size : int
        """
        if self.mask_type == 'random':
            if self.mask_attribute is None:
                num_visibles = np.random.randint(1, self.num_pixels, size=batch_size)
                return batch_random_mask(self.img_size, num_visibles, batch_size)
            elif type(self.mask_attribute) == int:
                return batch_random_mask(self.img_size, self.mask_attribute, batch_size)
            else:
                lower_bound, upper_bound = self.mask_attribute
                num_visibles = np.random.randint(lower_bound, upper_bound, size=batch_size)
                return batch_random_mask(self.img_size, num_visibles, batch_size)
        elif self.mask_type == 'bottom':
            return batch_bottom_mask(self.img_size, self.mask_attribute, batch_size)
        elif self.mask_type == 'top':
            return batch_top_mask(self.img_size, self.mask_attribute, batch_size)
        elif self.mask_type == 'center':
            return batch_center_mask(self.img_size, self.mask_attribute, batch_size)
        elif self.mask_type == 'edge':
            return batch_edge_mask(self.img_size, self.mask_attribute, batch_size)
        elif self.mask_type == 'left':
            return batch_left_mask(self.img_size, self.mask_attribute, batch_size)
        elif self.mask_type == 'right':
            return batch_right_mask(self.img_size, self.mask_attribute, batch_size)
        elif self.mask_type == 'random_rect':
            return batch_random_rect_mask(self.img_size, self.mask_attribute[0],
                                          self.mask_attribute[1], batch_size)
        elif self.mask_type == 'random_blob':
            return batch_multi_random_blobs(self.img_size,
                                            self.mask_attribute[0],
                                            self.mask_attribute[1],
                                            self.mask_attribute[2], batch_size)
        elif self.mask_type == 'random_blob_cache':
            # Hacky way to get a single batch of data
            for mask_batch in self.data_loader:
                break
            # Zero index because Image folder returns (img, label) tuple
            return mask_batch[0]


def single_random_mask(img_size, num_visible):
    """Returns random mask where 0 corresponds to a hidden value and 1 to a
    visible value. Shape of mask is same as img_size.

    Parameters
    ----------
    img_size : tuple of ints
        E.g. (1, 32, 32) for grayscale or (3, 64, 64) for RGB.

    num_visible : int
        Number of visible values.
    """
    _, height, width = img_size
    # Sample integers without replacement between 0 and the total number of
    # pixels. The measurements array will then contain a pixel indices
    # corresponding to locations where pixels will be visible.
    measurements = np.random.choice(range(height * width), size=num_visible, replace=False)
    # Create empty mask
    mask = torch.zeros(1, width, height)
    # Update mask with measurements
    for m in measurements:
        row = int(m / width)
        col = m % width
        mask[0, row, col] = 1
    return mask


def batch_random_mask(img_size, num_visibles, batch_size, repeat=False):
    """Returns a batch of random masks.

    Parameters
    ----------
    img_size : see single_random_mask

    num_visibles : int or list of ints
        If int will keep the number of visible pixels in the masks fixed, if
        list will change the number of visible pixels depending on the values
        in the list. List should have length equal to batch_size.

    batch_size : int
        Number of masks to create.

    repeat : bool
        If True returns a batch of the same mask repeated batch_size times.
    """
    # Mask should have same shape as image, but only 1 channel
    mask_batch = torch.zeros(batch_size, 1, *img_size[1:])
    if repeat:
        if not type(num_visibles) == int:
            raise RuntimeError("num_visibles must be an int if used with repeat=True. {} was provided instead.".format(type(num_visibles)))
        single_mask = single_random_mask(img_size, num_visibles)
        for i in range(batch_size):
            mask_batch[i] = single_mask
    else:
        if type(num_visibles) == int:
            for i in range(batch_size):
                mask_batch[i] = single_random_mask(img_size, num_visibles)
        else:
            for i in range(batch_size):
                mask_batch[i] = single_random_mask(img_size, num_visibles[i])
    return mask_batch


def batch_bottom_mask(img_size, num_rows, batch_size):
    """Masks all the output except the |num_rows| lowest rows (in the height
    dimension).

    Parameters
    ----------
    img_size : see single_random_mask

    num_rows : int
        Number of rows from bottom which will be visible.

    batch_size : int
        Number of masks to create.
    """
    mask = torch.zeros(batch_size, 1, *img_size[1:])
    mask[:, :, -num_rows:, :] = 1.
    return mask


def batch_top_mask(img_size, num_rows, batch_size):
    """Masks all the output except the |num_rows| highest rows (in the height
    dimension).

    Parameters
    ----------
    img_size : see single_random_mask

    num_rows : int
        Number of rows from top which will be visible.

    batch_size : int
        Number of masks to create.
    """
    mask = torch.zeros(batch_size, 1, *img_size[1:])
    mask[:, :, :num_rows, :] = 1.
    return mask


def batch_center_mask(img_size, num_pixels, batch_size):
    """Masks all the output except the num_pixels by num_pixels central square
    of the image.

    Parameters
    ----------
    img_size : see single_random_mask

    num_pixels : int
        Should be even. If not even, num_pixels will be replaced with
        num_pixels - 1.

    batch_size : int
        Number of masks to create.
    """
    mask = torch.zeros(batch_size, 1, *img_size[1:])
    _, height, width = img_size
    lower_height = int(height / 2 - num_pixels / 2)
    upper_height = int(height / 2 + num_pixels / 2)
    lower_width = int(width / 2 - num_pixels / 2)
    upper_width = int(width / 2 + num_pixels / 2)
    mask[:, :, lower_height:upper_height, lower_width:upper_width] = 1.
    return mask


def batch_edge_mask(img_size, num_pixels, batch_size):
    """Masks all the output except the num_pixels thick edge of the image.

    Parameters
    ----------
    img_size : see single_random_mask

    num_pixels : int
        Should be smaller than min(height / 2, width / 2).

    batch_size : int
        Number of masks to create.
    """
    mask = torch.zeros(batch_size, 1, *img_size[1:])
    mask[:, :, :num_pixels, :] = 1.
    mask[:, :, -num_pixels:, :] = 1.
    mask[:, :, :, :num_pixels] = 1.
    mask[:, :, :, -num_pixels:] = 1.
    return mask


def batch_left_mask(img_size, num_cols, batch_size):
    """Masks all the pixels except the left side of the image.

    Parameters
    ----------
    img_size : see single_random_mask

    num_cols : int
        Number of columns of the left side of the image to remain visible.

    batch_size : int
        Number of masks to create.
    """
    mask = torch.zeros(batch_size, 1, *img_size[1:])
    mask[:, :, :, :num_cols] = 1.
    return mask


def batch_right_mask(img_size, num_cols, batch_size):
    """Masks all the pixels except the right side of the image.

    Parameters
    ----------
    img_size : see single_random_mask

    num_cols : int
        Number of columns of the right side of the image to remain visible.

    batch_size : int
        Number of masks to create.
    """
    mask = torch.zeros(batch_size, 1, *img_size[1:])
    mask[:, :, :, -num_cols:] = 1.
    return mask


def random_rect_mask(img_size, max_height, max_width):
    """Returns a mask with a random rectangle of visible pixels.

    Parameters
    ----------
    img_size : see single_random_mask

    max_height : int
        Maximum height of randomly sampled rectangle.

    max_width : int
        Maximum width of randomly sampled rectangle.
    """
    mask = torch.zeros(1, *img_size[1:])
    _, img_width, img_height = img_size
    # Sample top left corner of unmasked rectangle
    top_left = np.random.randint(0, img_height - 1), np.random.randint(0, img_width - 1)
    # Sample height of rectangle
    # This is a number between 1 and the max_height parameter. If the top left corner
    # is too close to the bottom of the image, make sure the rectangle doesn't exceed
    # this
    rect_height = np.random.randint(1, min(max_height, img_height - top_left[0]))
    # Sample width of rectangle
    rect_width = np.random.randint(1, min(max_width, img_width - top_left[1]))
    # Set visible pixels
    bottom_right = top_left[0] + rect_height, top_left[1] + rect_width
    mask[0, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 1.
    return mask


def batch_random_rect_mask(img_size, max_height, max_width, batch_size):
    """Returns a batch of masks with random rectangles of visible pixels.

    Parameters
    ----------
    img_size : see single_random_mask

    max_height : int
        Maximum height of randomly sampled rectangle.

    max_width : int
        Maximum width of randomly sampled rectangle.

    batch_size : int
        Number of masks to create.
    """
    mask = torch.zeros(batch_size, 1, *img_size[1:])
    for i in range(batch_size):
        mask[i] = random_rect_mask(img_size, max_height, max_width)
    return mask


def random_blob(img_size, num_iter, threshold, fixed_init=None):
    """Generates masks with random connected blobs.

    Parameters
    ----------
    img_size : see single_random_mask

    num_iter : int
        Number of iterations to expand random blob for.

    threshold : float
        Number between 0 and 1. Probability of keeping a pixel hidden.

    fixed_init : tuple of ints or None
        If fixed_init is None, central position of blob will be sampled
        randomly, otherwise expansion will start from fixed_init. E.g.
        fixed_init = (6, 12) will start the expansion from pixel in row 6,
        column 12.
    """
    _, img_height, img_width = img_size
    # Defines the shifts around the central pixel which may be unmasked
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if fixed_init is None:
        # Sample random initial position
        init_pos = np.random.randint(0, img_height - 1), np.random.randint(0, img_width - 1)
    else:
        init_pos = (fixed_init[0], fixed_init[1])
    # Initialize mask and make init_pos visible
    mask = torch.zeros(1, 1, *img_size[1:])
    mask[0, 0, init_pos[0], init_pos[1]] = 1.
    # Initialize the list of seed positions
    seed_positions = [init_pos]
    # Randomly expand blob
    for i in range(num_iter):
        next_seed_positions = []
        for seed_pos in seed_positions:
            # Sample probability that neighboring pixel will be visible
            prob_visible = np.random.rand(len(neighbors))
            for j, neighbor in enumerate(neighbors):
                if prob_visible[j] > threshold:
                    current_h, current_w = seed_pos
                    shift_h, shift_w = neighbor
                    # Ensure new height stays within image boundaries
                    new_h = max(min(current_h + shift_h, img_height - 1), 0)
                    # Ensure new width stays within image boundaries
                    new_w = max(min(current_w + shift_w, img_width - 1), 0)
                    # Update mask
                    mask[0, 0, new_h, new_w] = 1.
                    # Add new position to list of seeds
                    next_seed_positions.append((new_h, new_w))
        seed_positions = next_seed_positions
    return mask


def multi_random_blobs(img_size, max_num_blobs, iter_range, threshold):
    """Generates masks with multiple random connected blobs.

    Parameters
    ----------
    max_num_blobs : int
        Maximum number of blobs. Number of blobs will be sampled between 1 and
        max_num_blobs

    iter_range : (int, int)
        Lower and upper bound on number of iterations to be used for each blob.
        This will be sampled for each blob.

    threshold : float
        Number between 0 and 1. Probability of keeping a pixel hidden.
    """
    mask = torch.zeros(1, 1, *img_size[1:])
    # Sample number of blobs
    num_blobs = np.random.randint(1, max_num_blobs + 1)
    for _ in range(num_blobs):
        num_iter = np.random.randint(iter_range[0], iter_range[1])
        mask += random_blob(img_size, num_iter, threshold)
    mask[mask > 0] = 1.
    return mask


def batch_multi_random_blobs(img_size, max_num_blobs, iter_range, threshold,
                             batch_size):
    """Generates batch of masks with multiple random connected blobs."""
    mask = torch.zeros(batch_size, 1, *img_size[1:])
    for i in range(batch_size):
        mask[i] = multi_random_blobs(img_size, max_num_blobs, iter_range, threshold)
    return mask


def get_conditional_pixels(batch, mask, num_colors):
    """Returns conditional pixels obtained from masking the data in batch with
    mask and appending the mask. E.g. if the input has size (N, C, H, W)
    then the output will have size (N, C + 1, H, W) i.e. the mask is appended
    as an extra color channel.

    Parameters
    ----------
    batch : torch.Tensor
        Batch of data as returned by a DataLoader, i.e. unnormalized.
        Shape (num_examples, num_channels, width, height)

    mask : torch.Tensor
        Mask as returned by MaskGenerator.get_masks.
        Shape (num_examples, 1, width, height)

    num_colors : int
        Number of colors image is quantized to.
    """
    batch_size, channels, width, height = batch.size()
    # Add extra channel to keep mask
    cond_pixels = torch.zeros((batch_size, channels + 1, height, width))
    # Mask batch to only show visible pixels
    cond_pixels[:, :channels, :, :] = mask * batch.float()
    # Add mask scaled by number of colors in last channel dimension
    cond_pixels[:, -1:, :, :] = mask * (num_colors - 1)
    # Normalize conditional pixels to be in 0 - 1 range
    return cond_pixels / (num_colors - 1)


def get_repeated_conditional_pixels(batch, mask, num_colors, num_reps):
    """Returns repeated conditional pixels.

    Parameters
    ----------
    batch : torch.Tensor
        Shape (1, num_channels, width, height)

    mask : torch.Tensor
        Shape (1, num_channels, width, height)

    num_colors : int
        Number of colors image is quantized to.

    num_reps : int
        Number of times the conditional pixels will be repeated
    """
    assert batch.size(0) == 1
    assert mask.size(0) == 1
    cond_pixels = get_conditional_pixels(batch, mask, num_colors)
    return cond_pixels.expand(num_reps, *cond_pixels.size()[1:])
