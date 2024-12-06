
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# This is the main training file for the first part of the assignment.
#
# Usage:
# ======
#    To train with the default hyperparamters
#    (saves results to checkpoints_vanilla/ and samples_vanilla/):
#       python vanilla_gan.py

import argparse
import os

import imageio
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss

import utils
from pathlib import Path
from models import DCGenerator, DCDiscriminator
from dataset import TerrainDataset


from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Dataset
from torch.linalg import vector_norm


from PIL import Image
from pathlib import Path
from tqdm import tqdm


policy = 'color,translation,cutout'

SEED = 11

# Set the random seed manually for reproducibility.
# torch.set_default_device('cuda:1')

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_models(G, D):
    """Prints model information for the generators and discriminators.
    """
    print("                    G                  ")
    print("---------------------------------------")
    print(G)
    print("---------------------------------------")

    print("                    D                  ")
    print("---------------------------------------")
    print(D)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators.
    """
    G = DCGenerator()
    D = DCDiscriminator()

    print_models(G, D)

    if torch.cuda.is_available():
        G.cuda("cuda:2")
        D.cuda("cuda:2")
        print('Models moved to GPU.')

    return G, D


def create_image_grid(array, ncols=None):
    """Useful docstring (insert there)."""
    num_images, channels, cell_h, cell_w = array.shape

    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros(
        (cell_h * nrows, cell_w * ncols, channels),
        dtype=array.dtype
    )
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[
                i * cell_h:(i + 1) * cell_h,
                j * cell_w:(j + 1) * cell_w, :
            ] = array[i * ncols + j].transpose(1, 2, 0)

    if channels == 1:
        result = result.squeeze()
    return result


def checkpoint(iteration, G, D, opts):
    """Save the parameters of the generator G and discriminator D."""
    G_path = os.path.join(opts.checkpoint_dir, 'G_iter%d.pkl' % iteration)
    # D_path = os.path.join(opts.checkpoint_dir, 'D_iter%d.pkl' % iteration)
    torch.save(G.state_dict(), G_path)
    # torch.save(D.state_dict(), D_path)


def save_samples(G, fixed_noise, iteration, opts):
    generated_images = G(fixed_noise)
    generated_images = utils.to_data(generated_images)

    grid = create_image_grid(generated_images)
    grid = np.uint8(255 * (grid + 1) / 2)

    # merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}.png'.format(iteration))
    imageio.imwrite(path, grid)
    print('Saved {}'.format(path))


def save_images(images, iteration, opts, name):
    grid = create_image_grid(utils.to_data(images))

    path = os.path.join(
        opts.sample_dir,
        '{:s}-{:06d}.png'.format(name, iteration)
    )
    grid = np.uint8(255 * (grid + 1) / 2)
    imageio.imwrite(path, grid)
    print('Saved {}'.format(path))


def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Variable of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Variable of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return utils.to_var(
        torch.rand(batch_size, dim) * 2 - 1
    ).unsqueeze(2).unsqueeze(3)


def training_loop(opts):
    """Runs the training loop.
        * Saves checkpoints every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """
    
    ### Setup Dataloader
    normal_directory = Path("data/rugd")
    blurry_directory = Path("blurry_images")
    
    transform = transforms.Compose([
        # transforms.RandomCrop((256, 256)),
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.ColorJitter(),
        # transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = TerrainDataset(normal_directory, blurry_directory, transform=transform)

    # Create DataLoader objects for each subset
    train_dataloader = DataLoader(dataset, batch_size=opts.batch_size, num_workers=4, pin_memory=True, shuffle=True)

    # Create generators and discriminators
    G, D = create_model(opts)

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(G.parameters(), opts.lr * 10, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])

    iteration = 1

    total_train_iters = opts.num_epochs * len(train_dataloader)
    
    # criterion = BCEWithLogitsLoss()
    criterion = torch.nn.MSELoss()
    criterion_img = torch.nn.MSELoss()
    
    G.train()
    D.train()
    
    for _ in range(opts.num_epochs):

        for batch in train_dataloader:

            original, blurry = batch
            batch_size = len(original)
            original = utils.to_var(original)
            blurry = utils.to_var(blurry)
            
            # save_images(original, iteration, opts, 'original')
            # save_images(blurry, iteration, opts, 'blurry')

            # exit()
            
            # TRAIN THE DISCRIMINATOR
            # D.train()
            # G.eval()
            fake_images = G(blurry)
            
            # print(torch.ones(batch_size, 1).shape)
            # print(batch_size)
            # print(original.shape)
            # print(blurry.shape)
            # print(D(original).shape)
            
            discriminator_prediction_real = D(original)  # Real: blurry + original
            discriminator_prediction_fake = D(fake_images)  # Fake: blurry + fake

            D_real_loss = criterion(discriminator_prediction_real, utils.to_var(torch.ones(batch_size, 1)))
            D_fake_loss = criterion(discriminator_prediction_fake, utils.to_var(torch.zeros(batch_size, 1)))
            D_total_loss = (D_real_loss + D_fake_loss) * 0.075

            d_optimizer.zero_grad()
            D_total_loss.backward(retain_graph=True)
            d_optimizer.step()

            # TRAIN THE GENERATOR
            # G.train()
            # D.eval()
            fake_or_not = D(fake_images)  # Evaluate updated fake images

            lambda_reconstruction = 0.2
            adversarial_loss = criterion(fake_or_not, utils.to_var(torch.ones(batch_size, 1)))  # Adversarial loss
            pixel_loss = torch.mean(vector_norm(fake_images - original) ** 2)  # Reconstruction loss
            G_loss = lambda_reconstruction * adversarial_loss + pixel_loss

            g_optimizer.zero_grad()
            G_loss.backward()
            g_optimizer.step()

            # Print the log info
            if iteration % opts.log_step == 0:
                print(
                    'Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | '
                    'D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                        iteration, total_train_iters, D_real_loss.item(),
                        D_fake_loss.item(), G_loss.item()
                    )
                )
                logger.add_scalar('D/fake', D_fake_loss, iteration)
                logger.add_scalar('D/real', D_real_loss, iteration)
                logger.add_scalar('D/total', D_total_loss, iteration)
                logger.add_scalar('G/total', G_loss, iteration)

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                # save_samples(G, fixed_noise, iteration, opts)
                # save_images(original, iteration, opts, 'original')
                # save_images(blurry, iteration, opts, 'blurry')
                save_images(fake_images, iteration, opts, 'samples')

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1


def main(opts):
    """Loads the data and starts the training loop."""
    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    training_loop(opts)    


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--data', type=str, default='cat/grumpifyBprocessed')
    parser.add_argument('--data_preprocess', type=str, default='basic')
    parser.add_argument('--ext', type=str, default='*.png')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', default='./checkpoints_vanilla')
    parser.add_argument('--sample_dir', type=str, default='./vanilla')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--checkpoint_every', type=int, default=400)

    return parser


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size
    opts.sample_dir = os.path.join(
        'output/', opts.sample_dir,
        '%s_%s' % (os.path.basename(opts.data), opts.data_preprocess)
    )

    if os.path.exists(opts.sample_dir):
        cmd = 'rm %s/*' % opts.sample_dir
        os.system(cmd)
    logger = SummaryWriter(opts.sample_dir)
    print(opts)
    main(opts)
