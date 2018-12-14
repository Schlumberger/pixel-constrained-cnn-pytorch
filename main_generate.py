import argparse
import json
import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
from pixconcnn.generate import generate_images
from utils.loading import load_model
from utils.masks import get_conditional_pixels
from utils.plots import uncertainty_plot, probs_and_conditional_plot
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model-folder', dest='model_folder', default=None,
                    help='Path to trained pixel constrained model folder',
                    required=True)
parser.add_argument('-n', '--name', dest='name', default=None,
                    help='Name of generation experiment', required=True)
parser.add_argument('-t', '--gen_type', dest='gen_type', default=None,
                    choices=['generation','logits','uncertainty'],
                    help='Type of generation', required=True)
parser.add_argument('-i', '--imgs', dest='imgs_idx', default=None,
                    type=int, nargs='+',
                    help='List of indices of images to perform generation with.')
parser.add_argument('-ns', '--num-samples', dest='num_samples', default=None,
                    type=int, help='Number of samples to generate for each image-mask combo.',
                    required=True)
parser.add_argument('-te', '--temp', dest='temp', default=1.,
                    type=float, help='Sampling temperature.')
parser.add_argument('-v', '--model-version', dest='model_version', default=None,
                    type=int, help='Version of model if not using the latest one.')

parser.add_argument('-nr', '--num-row', dest='num_per_row', default=8,
                    type=int, help='Number of images per row in grids.')
parser.add_argument('-ni', '--num-iterations', dest='num_iters', default=None,
                    type=int, help='Only relevant for logits. Number of iterations to plot intermediate logits for.')
parser.add_argument('-r', '--random', dest='random_attribute', default=None,
                    type=int, help='Number of random pixels to keep unmasked.')
parser.add_argument('-b', '--bottom', dest='bottom_attribute', default=None,
                    type=int, help='Number of bottom pixels to keep unmasked.')
parser.add_argument('-to', '--top', dest='top_attribute', default=None,
                    type=int, help='Number of top pixels to keep unmasked.')
parser.add_argument('-c', '--center', dest='center_attribute', default=None,
                    type=int, help='Number of central pixels to keep unmasked.')
parser.add_argument('-e', '--edge', dest='edge_attribute', default=None,
                    type=int, help='Number of edge pixels to keep unmasked.')
parser.add_argument('-l', '--left', dest='left_attribute', default=None,
                    type=int, help='Number of left pixels to keep unmasked.')
parser.add_argument('-ri', '--right', dest='right_attribute', default=None,
                    type=int, help='Number of right pixels to keep unmasked.')
parser.add_argument('-rb', '--random-blob', dest='blob_attribute', default=None,
                    type=int, nargs='+', help='First int should be maximum number of blobs, second lower bound on num_iters and third upper bound on num_iters.')
parser.add_argument('-mf', '--mask-folder', dest='folder_attribute', default=None,
                    help='Mask folder if using a cached mask.')

# Unpack args
args = parser.parse_args()

# Create a folder to store generation results
timestamp = time.strftime("%Y-%m-%d_%H-%M")
directory = "gen_{}_{}".format(timestamp, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)

# Save args
with open(directory + '/args.json', 'w') as args_file:
    json.dump(vars(args), args_file)

# Load model
model, data_loader, _ = load_model(args.model_folder, model_version=args.model_version)

# Convert input arguments to mask_descriptors
mask_descriptors = []
if args.random_attribute is not None:
    mask_descriptors.append(('random', args.random_attribute))
if args.bottom_attribute is not None:
    mask_descriptors.append(('bottom', args.bottom_attribute))
if args.top_attribute is not None:
    mask_descriptors.append(('top', args.top_attribute))
if args.center_attribute is not None:
    mask_descriptors.append(('center', args.center_attribute))
if args.edge_attribute is not None:
    mask_descriptors.append(('edge', args.edge_attribute))
if args.left_attribute is not None:
    mask_descriptors.append(('left', args.left_attribute))
if args.right_attribute is not None:
    mask_descriptors.append(('right', args.right_attribute))
if args.blob_attribute is not None:
    max_num_blobs, lower_iter, upper_iter = args.blob_attribute
    mask_descriptors.append(('random_blob', (max_num_blobs, (lower_iter, upper_iter), 0.5)))
if args.folder_attribute is not None:
    mask_descriptors.append(('random_blob_cache', (args.folder_attribute, 1)))

imgs_idx = args.imgs_idx
num_img = len(imgs_idx)
total_num_imgs = args.num_samples * num_img * len(mask_descriptors)
print("\nGenerating {} samples for {} different images combined with {} masks for a total of {} images".format(args.num_samples, num_img, len(mask_descriptors), total_num_imgs))
print("\nThe masks are {}\n".format(mask_descriptors))

# Create a batch from the images in imgs_idx
batch = torch.stack([data_loader.dataset[img_idx][0] for img_idx in imgs_idx], dim=0)

if args.gen_type == 'generation' or args.gen_type == 'uncertainty':
    # Generate images with model
    outputs = generate_images(model, batch, mask_descriptors,
                              num_samples=args.num_samples, temp=args.temp,
                              verbose=True)

    # Save images in folder
    for i in range(num_img):
        for j in range(len(mask_descriptors)):
            output = outputs[i][j]
            # Save every output as an image and a pytorch tensor
            torch.save(output["orig_img"].cpu(), directory + "/source_{}_{}.pt".format(i, j))
            save_image(output["orig_img"].float() / (model.prior_net.num_colors - 1.), directory + "/source_{}_{}.png".format(i, j))
            torch.save(output["cond_pixels"][0:1].cpu(), directory + "/cond_pixels_{}_{}.pt".format(i, j))
            save_image(output["cond_pixels"][0:1, :3], directory + "/cond_pixels_{}_{}.png".format(i, j))
            torch.save(output["mask"].cpu(), directory + "/mask_{}_{}.pt".format(i, j))
            save_image(output["mask"], directory + "/mask_{}_{}.png".format(i, j))
            save_image(output["samples"].float().mean(dim=0) / (model.prior_net.num_colors - 1.), directory + '/mean_samples_{}_{}.png'.format(i, j))
            torch.save(output["samples"].cpu(), directory + "/samples_{}_{}.pt".format(i, j))
            torch.save(output["log_probs"].cpu(), directory + "/log_probs_{}_{}.pt".format(i, j))
            if args.gen_type == 'generation':
                save_image(output["samples"].float() / (model.prior_net.num_colors - 1.), directory + "/samples_{}_{}.png".format(i, j), nrow=args.num_per_row, pad_value=1)
            elif args.gen_type == 'uncertainty':
                sorted_samples, log_likelihoods = uncertainty_plot(output["samples"], output["log_probs"])
                save_image(sorted_samples.float() / (model.prior_net.num_colors - 1.), directory + "/sorted_samples_{}_{}.png".format(i, j), nrow=args.num_per_row, pad_value=1)
                save_image(log_likelihoods, directory + "/log_likelihoods_{}_{}.png".format(i, j), pad_value=1, nrow=args.num_per_row)
elif args.gen_type == 'logits': # Note this only works for binary images
    if model.prior_net.num_colors != 2:
        raise(RuntimeError("Logits generation only works for models with 2 colors. Current model has {} colors.".format(model.prior_net.num_colors)))
    # Generate images with model
    outputs = generate_images(model, batch, mask_descriptors,
                              num_samples=args.num_samples, verbose=True,
                              temp=args.temp)
    # Extract info
    img_size = model.prior_net.img_size
    num_pixels = img_size[1] * img_size[2]
    pix_per_iters = num_pixels // args.num_iters  # Number of pixels to unmask per iteration
    # Save images in folder
    for i in range(num_img):
        for j in range(len(mask_descriptors)):
            output = outputs[i][j]
            mask = output["mask"]
            mask = mask.expand(args.num_samples, *mask.size()[1:])
            samples = output["samples"]
            cond_pixels = get_conditional_pixels(samples, mask.float(), 2)
            cond_pixels = cond_pixels.to(device)
            mask = mask.to(device)
            samples = samples.to(device)
            logit_plots = {}
            for k in range(args.num_iters):
                logit_plots[k] = {}
                mask_step = mask.clone()
                if k != 0:  # Do not modify mask for first iteration
                    # Unmask num_pix_to_unmask pixels in raster order
                    num_pix_to_unmask = k * pix_per_iters
                    num_rows = num_pix_to_unmask // img_size[2]
                    num_cols = num_pix_to_unmask - num_rows * img_size[2]
                    if num_rows != 0:
                        mask_step[:, :, :num_rows, :] = 1
                    if num_cols != 0:
                        mask_step[:, :, num_rows, :num_cols] = 1
                # Calculate logits with updated mask
                logits, prior_logits, cond_logits = model(samples.float() * mask_step.float(), cond_pixels)
                probs = F.softmax(logits.detach(), dim=1)
                # Create a plot for each sample
                for l in range(args.num_samples):
                    logit_plot = probs_and_conditional_plot(samples[l].cpu(),
                                                            probs[l, 1, 0].cpu(),
                                                            mask_step[0, 0].cpu())
                    logit_plots[k][l] = logit_plot

            # Plot iterations as a grid for each sample
            for l in range(args.num_samples):
                logit_grid = []
                for k in range(args.num_iters):
                    logit_grid.append(logit_plots[k][l])
                stacked_images = torch.stack(logit_grid, dim=0)
                save_image(stacked_images, directory + "/logit_plot_{}_{}_{}.png".format(i, j, l), pad_value=1)
                torch.save(stacked_images.cpu(), directory + "/logit_plot_{}_{}_{}.pt".format(i, j, l))
