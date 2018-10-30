import imageio
import json
import os
import sys
import time
import torch
from pixconcnn.training import Trainer, PixelConstrainedTrainer
from torchvision.utils import save_image
from utils.dataloaders import mnist, celeba
from utils.init_models import initialize_model
from utils.masks import batch_random_mask, get_repeated_conditional_pixels, MaskGenerator


# Set device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Get config file from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python main.py <path_to_config>"))
config_path = sys.argv[1]

# Open config file
with open(config_path) as config_file:
    config = json.load(config_file)

name = config['name']
constrained = config['constrained']
batch_size = config['batch_size']
lr = config['lr']
num_colors = config['num_colors']
epochs = config['epochs']
dataset = config['dataset']
resize = config['resize']  # Only relevant for celeba
crop = config['crop']  # Only relevant for celeba
grayscale = config["grayscale"]  # Only relevant for celeba
num_conds = config['num_conds']  # Only relevant if constrained
num_samples = config['num_samples']  # Only relevant if constrained
filter_size = config['filter_size']
depth = config['depth']
num_filters_cond = config['num_filters_cond']
num_filters_prior = config['num_filters_prior']
mask_descriptor = config['mask_descriptor']
weight_cond_logits_loss = config['weight_cond_logits_loss']
weight_prior_logits_loss = config['weight_prior_logits_loss']

# Create a folder to store experiment results
timestamp = time.strftime("%Y-%m-%d_%H-%M")
directory = "{}_{}".format(timestamp, name)
if not os.path.exists(directory):
    os.makedirs(directory)

# Save config file in experiment directory
with open(directory + '/config.json', 'w') as config_file:
    json.dump(config, config_file)

# Get data
if dataset == 'mnist':
    data_loader, _ = mnist(batch_size, num_colors=num_colors, size=resize)
    img_size = (1, resize, resize)
elif dataset == 'celeba':
    data_loader = celeba(batch_size, num_colors=num_colors, size=resize,
                         crop=crop, grayscale=grayscale)
    if grayscale:
        img_size = (1, resize, resize)
    else:
        img_size = (3, resize, resize)

# Initialize model weights and architecture
model = initialize_model(img_size,
                         num_colors,
                         depth,
                         filter_size,
                         constrained,
                         num_filters_prior,
                         num_filters_cond)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if constrained:
    mask_generator = MaskGenerator(img_size, mask_descriptor)
    trainer = PixelConstrainedTrainer(model, optimizer, device, mask_generator,
                                      weight_cond_logits_loss=weight_cond_logits_loss,
                                      weight_prior_logits_loss=weight_prior_logits_loss)
    # Train model
    progress_imgs = trainer.train(data_loader, epochs, directory=directory)

    # Get a random batch of images
    for batch, _ in data_loader:
        break

    for i in range(num_conds):
        mask = mask_generator.get_masks(batch_size)
        print('Generating {}/{} conditionings'.format(i + 1, num_conds))
        cond_pixels = get_repeated_conditional_pixels(batch[i:i+1], mask[i:i+1],
                                                      num_colors, num_samples)
        # Save mask as tensor
        torch.save(mask[i:i+1], directory + '/mask{}.pt'.format(i))
        # Save image that gave rise to the conditioning as tensor
        torch.save(batch[i:i+1], directory + '/source{}.pt'.format(i))
        # Save conditional pixels as tensor and image
        torch.save(cond_pixels[0:1], directory + '/cond_pixels{}.pt'.format(i))
        save_image(cond_pixels[0:1], directory + '/cond_pixels{}.png'.format(i))

        cond_pixels = cond_pixels.to(device)
        samples = model.sample(cond_pixels)
        # Save samples and mean sample as tensor and image
        torch.save(samples, directory + '/samples_cond{}.pt'.format(i))
        save_image(samples.float() / (num_colors - 1.),
                   directory + '/samples_cond{}.png'.format(i))
        save_image(samples.float().mean(dim=0) / (num_colors - 1.),
                   directory + '/mean_cond{}.png'.format(i))
        # Save conditional logits if image is binary
        if num_colors == 2:
            # Save conditional logits
            logits, _, cond_logits = model(batch[i:i+1].float().to(device), cond_pixels[0:1])
            # Second dimension corresponds to different pixel values, so select probs of it being 1
            save_image(cond_logits[:, 1], directory + '/prob_of_one_cond{}.png'.format(i))
            # Second dimension corresponds to different pixel values, so select probs of it being 1
            save_image(logits[:, 1], directory + '/prob_of_one_logits{}.png'.format(i))
else:
    trainer = Trainer(model, optimizer, device)
    progress_imgs = trainer.train(data_loader, epochs, directory=directory)

# Save losses and plots of them
with open(directory + '/losses.json', 'w') as losses_file:
    json.dump(trainer.losses, losses_file)

# Save model
torch.save(trainer.model.state_dict(), directory + '/model.pt')

# Save gif of progress
imageio.mimsave(directory + '/training.gif', progress_imgs, fps=24)
