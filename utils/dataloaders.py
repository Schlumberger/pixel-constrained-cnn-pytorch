import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def mnist(batch_size=128, num_colors=256, size=28,
          path_to_data='../mnist_data'):
    """MNIST dataloader with (28, 28) images.

    Parameters
    ----------
    batch_size : int

    num_colors : int
        Number of colors to quantize images into. Typically 256, but can be
        lower for e.g. binary images.

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    quantize = get_quantize_func(num_colors)

    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: quantize(x))
    ])

    train_data = datasets.MNIST(path_to_data, train=True, download=False,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def celeba(batch_size=128, num_colors=256, size=64, crop=64, grayscale=False,
           shuffle=True, path_to_data='../celeba_64'):
    """CelebA dataloader with (64, 64) images.

    Parameters
    ----------
    batch_size : int

    num_colors : int
        Number of colors to quantize images into. Typically 256, but can be
        lower for e.g. binary images.

    size : int
        Size (height and width) of each image. Default is 64 for no resizing.

    crop : int
        Size of center crop. This crop happens *before* the resizing.

    grayscale : bool
        If True converts images to grayscale.

    shuffle : bool
        If True shuffles images.

    path_to_data : string
        Path to 64 by 64 CelebA data files.
    """
    quantize = get_quantize_func(num_colors)

    if grayscale:
        transform = transforms.Compose([
            transforms.CenterCrop(crop),
            transforms.Resize(size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: quantize(x))
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(crop),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: quantize(x))
        ])
    celeba_data = CelebADataset(path_to_data,
                                transform=transform)
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=shuffle)
    return celeba_loader


class CelebADataset(Dataset):
    """CelebA dataset with 64 by 64 images.

    Parameters
    ----------
    path_to_data : string
        Path to 64 by 64 CelebA data files.

    subsample : int
        Only load every |subsample| number of images.

    transform : None or one of torchvision.transforms instances
    """
    def __init__(self, path_to_data, subsample=1, transform=None):
        self.img_paths = glob.glob(path_to_data + '/*')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = Image.open(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, return 0 for the "label"
        return sample, 0


def get_quantize_func(num_colors):
    """Returns a quantization function which can be used to set the number of
    colors in an image.

    Parameters
    ----------
    num_colors : int
        Number of bins to quantize image into. Should be between 2 and 256.
    """
    def quantize_func(batch):
        """Takes as input a float tensor with values in the 0 - 1 range and
        outputs a long tensor with integer values corresponding to each
        quantization bin.

        Parameters
        ----------
        batch : torch.Tensor
            Values in 0 - 1 range.
        """
        if num_colors == 2:
            return (batch > 0.5).long()
        else:
            return (batch * (num_colors - 1)).long()

    return quantize_func
