import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masks import get_conditional_pixels
from torchvision.utils import make_grid


class Trainer():
    """Class used to train PixelCNN models without conditioning.

    Parameters
    ----------
    model : pixconcnn.models.gated_pixelcnn.GatedPixelCNN(RGB) instance

    optimizer : one of optimizers in torch.optim

    device : torch.device instance

    record_loss_every : int
        Frequency (in iterations) with which to record loss.

    save_model_every : int
        Frequency (in epochs) with which to save model.
    """
    def __init__(self, model, optimizer, device, record_loss_every=10,
                 save_model_every=5):
        self.device = device
        self.losses = {'total': []}
        self.mean_epoch_losses = []
        self.model = model
        self.optimizer = optimizer
        self.record_loss_every = record_loss_every
        self.save_model_every = save_model_every
        self.steps = 0

    def train(self, data_loader, epochs, directory='.'):
        """Trains model on the data given in data_loader.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader instance

        epochs : int
            Number of epochs to train model for.

        directory : string
            Directory in which to store training progress, including trained
            models and samples generated at every epoch.

        Returns
        -------
        List of numpy arrays of generated images after each epoch.
        """
        # List of generated images after each epoch to track progress of model
        progress_imgs = []

        for epoch in range(epochs):
            print("\nEpoch {}/{}".format(epoch + 1, epochs))
            epoch_loss = self._train_epoch(data_loader)
            mean_epoch_loss = epoch_loss / len(data_loader)
            print("Epoch loss: {}".format(mean_epoch_loss))
            self.mean_epoch_losses.append(mean_epoch_loss)

            # Create a grid of model samples (limit number of samples by scaling
            # by number of pixels in output image; this is needed because of
            # GPU memory limitations)
            if self.model.img_size[-1] > 32:
                scale_to_32 = self.model.img_size[-1] / 32
                num_images = 64 / (scale_to_32 * scale_to_32)
            else:
                num_images = 64
            # Generate samples from model
            samples = self.model.sample(self.device, num_images)
            img_grid = make_grid(samples).cpu()
            # Convert to numpy with channels in imageio order
            img_grid = img_grid.float().numpy().transpose(1, 2, 0) / (self.model.num_colors - 1.)
            progress_imgs.append(img_grid)
            # Save generated image
            imageio.imsave(directory + '/training{}.png'.format(epoch), progress_imgs[-1])
            # Save model
            if epoch % self.save_model_every == 0:
                torch.save(self.model.state_dict(),
                           directory + '/model{}.pt'.format(epoch))

        return progress_imgs

    def _train_epoch(self, data_loader):
        epoch_loss = 0
        for i, (batch, _) in enumerate(data_loader):
            batch_loss = self._train_iteration(batch)
            epoch_loss += batch_loss
            if i % 50 == 0:
                print("Iteration {}/{}, Loss: {}".format(i + 1,
                                                         len(data_loader),
                                                         batch_loss))
        return epoch_loss

    def _train_iteration(self, batch):
        self.optimizer.zero_grad()

        batch = batch.to(self.device)

        # Normalize batch, i.e. put it in 0 - 1 range before passing it through
        # the model
        norm_batch = batch.float() / (self.model.num_colors - 1)
        logits = self.model(norm_batch)

        loss = self._loss(logits, batch)
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        return loss.item()

    def _loss(self, logits, batch):
        loss = F.cross_entropy(logits, batch)

        if self.steps % self.record_loss_every == 0:
            self.losses['total'].append(loss.item())

        return loss


class PixelConstrainedTrainer():
    """Class used to train Pixel Constrained CNN models.

    Parameters
    ----------
    model : pixconcnn.models.pixel_constrained.PixelConstrained instance

    optimizer : one of optimizers in torch.optim

    device : torch.device instance

    mask_generator : pixconcnn.utils.masks.MaskGenerator instance
        Defines the masks used during training.

    weight_cond_logits_loss : float
        Weight on conditional logits in the loss (called alpha in the paper)

    weight_cond_logits_loss : float
        Weight on prio logits in the loss.

    record_loss_every : int
        Frequency (in iterations) with which to record loss.

    save_model_every : int
        Frequency (in epochs) with which to save model.
    """
    def __init__(self, model, optimizer, device, mask_generator,
                 weight_cond_logits_loss=0., weight_prior_logits_loss=0.,
                 record_loss_every=10, save_model_every=5):
        self.device = device
        self.losses = {'cond_logits': [], 'prior_logits': [], 'logits': [], 'total': []}  # Keep track of losses
        self.mask_generator = mask_generator
        self.mean_epoch_losses = []
        self.model = model
        self.optimizer = optimizer
        self.record_loss_every = record_loss_every
        self.save_model_every = save_model_every
        self.steps = 0
        self.weight_cond_logits_loss = weight_cond_logits_loss
        self.weight_prior_logits_loss = weight_prior_logits_loss

    def train(self, data_loader, epochs, directory='.'):
        """
        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader instance

        epochs : int
            Number of epochs to train model for.

        directory : string
            Directory in which to store training progress, including trained
            models and samples generated at every epoch.

        Returns
        -------
        List of numpy arrays of generated images after each epoch.
        """
        # List of generated images after each epoch to track progress of model
        progress_imgs = []

        # Use a fixed batch of images to test conditioning throughout training
        for batch, _ in data_loader:
            break
        test_mask = self.mask_generator.get_masks(batch.size(0))
        cond_pixels = get_conditional_pixels(batch, test_mask,
                                             self.model.prior_net.num_colors)
        # Number of images generated in a batch is limited by GPU memory.
        # For 32 by 32 this should be 64, for 64 by 64 this should be 16 and
        # for 128 by 128 this should be 4 etc.
        if self.model.prior_net.img_size[-1] > 32:
            scale_to_32 = self.model.prior_net.img_size[-1] / 32
            num_images = 64 / (scale_to_32 * scale_to_32)
        else:
            num_images = 64
        cond_pixels = cond_pixels[:num_images]

        cond_pixels = cond_pixels.to(self.device)

        for epoch in range(epochs):
            print("\nEpoch {}/{}".format(epoch + 1, epochs))
            epoch_loss = self._train_epoch(data_loader)
            mean_epoch_loss = epoch_loss / len(data_loader)
            print("Epoch loss: {}".format(mean_epoch_loss))
            self.mean_epoch_losses.append(mean_epoch_loss)

            # Create a grid of model samples
            samples = self.model.sample(cond_pixels)
            img_grid = make_grid(samples, nrow=8).cpu()
            # Convert to numpy with channels in imageio order
            img_grid = img_grid.float().numpy().transpose(1, 2, 0) / (self.model.prior_net.num_colors - 1.)
            progress_imgs.append(img_grid)

            # Save generated image
            imageio.imsave(directory + '/training{}.png'.format(epoch), progress_imgs[-1])

            # Save model
            if epoch % self.save_model_every == 0:
                torch.save(self.model.state_dict(),
                           directory + '/model{}.pt'.format(epoch))

        return progress_imgs

    def _train_epoch(self, data_loader):
        epoch_loss = 0
        for i, (batch, _) in enumerate(data_loader):
            mask = self.mask_generator.get_masks(batch.size(0))
            batch_loss = self._train_iteration(batch, mask)
            epoch_loss += batch_loss
            if i % 50 == 0:
                print("Iteration {}/{}, Loss: {}".format(i + 1, len(data_loader),
                                                         batch_loss))
        return epoch_loss

    def _train_iteration(self, batch, mask):
        self.optimizer.zero_grad()

        # Note that towards the end of the dataset the batch size may be smaller
        # if batch_size doesn't divide the number of examples. In that case,
        # slice mask so it has same shape as batch.
        cond_pixels = get_conditional_pixels(batch, mask[:batch.size(0)], self.model.prior_net.num_colors)

        batch = batch.to(self.device)
        cond_pixels = cond_pixels.to(self.device)

        # Normalize batch, i.e. put it in 0 - 1 range before passing it through
        # the model
        norm_batch = batch.float() / (self.model.prior_net.num_colors - 1)
        logits, prior_logits, cond_logits = self.model(norm_batch, cond_pixels)

        loss = self._loss(logits, prior_logits, cond_logits, batch)
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        return loss.item()

    def _loss(self, logits, prior_logits, cond_logits, batch):
        logits_loss = F.cross_entropy(logits, batch)
        prior_logits_loss = F.cross_entropy(prior_logits, batch)
        cond_logits_loss = F.cross_entropy(cond_logits, batch)
        total_loss = logits_loss + \
                     self.weight_cond_logits_loss * cond_logits_loss + \
                     self.weight_prior_logits_loss * prior_logits_loss

        # Record losses
        if self.steps % self.record_loss_every == 0:
            self.losses['total'].append(total_loss.item())
            self.losses['cond_logits'].append(cond_logits_loss.item())
            self.losses['prior_logits'].append(prior_logits_loss.item())
            self.losses['logits'].append(logits_loss.item())

        return total_loss
