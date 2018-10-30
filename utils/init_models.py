from pixconcnn.models.cnn import ResNet
from pixconcnn.models.gated_pixelcnn import GatedPixelCNN, GatedPixelCNNRGB
from pixconcnn.models.pixel_constrained import PixelConstrained


def initialize_model(img_size, num_colors, depth, filter_size, constrained,
                     num_filters_prior, num_filters_cond=1):
    """Helper function that initializes an appropriate model based on the
    input arguments.

    Parameters
    ----------
    img_size : tuple of ints
        Specifies size of image as (channels, height, width), e.g. (3, 32, 32).
        If img_size[0] == 1, returned model will be for grayscale images. If
        img_size[0] == 3, returned model will be for RGB images.

    num_colors : int
        Number of colors to quantize output into. Typically 256, but can be
        lower for e.g. binary images.

    depth : int
        Number of layers in model.

    filter_size : int
        Size of (square) convolutional filters of the model.

    constrained : bool
        If True returns a PixelConstrained model, otherwise returns a
        GatedPixelCNN or GatedPixelCNNRGB model.

    num_filters_prior : int
        Number of convolutional filters in each layer of prior network.

    num_filter_cond : int (optional)
        Required if using a PixelConstrained model. Number of of convolutional
        filters in each layer of conditioning network.
    """

    if img_size[0] == 1:
        prior_net = GatedPixelCNN(img_size=img_size,
                                  num_colors=num_colors,
                                  num_filters=num_filters_prior,
                                  depth=depth,
                                  filter_size=filter_size)
    else:
        prior_net = GatedPixelCNNRGB(img_size=img_size,
                                     num_colors=num_colors,
                                     num_filters=num_filters_prior,
                                     depth=depth,
                                     filter_size=filter_size)

    if constrained:
        # Add extra color channel for mask in conditioning network
        cond_net = ResNet(img_size=(img_size[0] + 1,) + img_size[1:],
                          num_colors=num_colors,
                          num_filters=num_filters_cond,
                          depth=depth,
                          filter_size=filter_size)
        # Define a pixel constrained model based on prior and cond net
        return PixelConstrained(prior_net, cond_net)
    else:
        return prior_net
