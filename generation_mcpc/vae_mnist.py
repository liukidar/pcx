# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F

import tempfile, shutil, os, subprocess, warnings
from pytorch_fid.fid_score_mnist import save_stats_mnist, fid_mnist
from inception_score import get_mnist_inception_score

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import uuid
import os
import argparse

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, activation):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc41 = nn.Linear(h_dim, z_dim)
        self.fc42 = nn.Linear(h_dim, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, h_dim)
        self.fc6 = nn.Linear(h_dim, h_dim)
        self.fc7 = nn.Linear(h_dim, x_dim)

        self.act_fn = activation
        
    def encoder(self, x):
        h = self.act_fn(self.fc1(x))
        h = self.act_fn(self.fc2(h))
        h = self.act_fn(self.fc3(h))
        return self.fc41(h), self.fc42(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = self.act_fn(self.fc4(z))
        h = self.act_fn(self.fc5(h))
        h = self.act_fn(self.fc6(h))
        return torch.sigmoid(self.fc7(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(epoch, vae, train_loader, optimizer, loss_function, verbose):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            if verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() / len(data)))
    if verbose:
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def tmp_save_imgs(imgs):
    tf = tempfile.NamedTemporaryFile()
    new_folder = False
    while not new_folder:
        try:
            new_folder=True
            os.makedirs("./data"+tf.name+"_")
        except OSError:
            print("ERROR")
            tf = tempfile.NamedTemporaryFile()
            new_folder=False
    for img_idx in range(len(imgs)):
        save_image(imgs[img_idx], "./data"+tf.name+"_"+"/"+str(img_idx)+".png")
    return "./data"+tf.name+"_"


def make_compressed_MNIST_files(test_dataset, data_folder):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    # save test images
    test_img_folder = data_folder + "/mnist_test"
    data, label = list(test_loader)[0]
    images = data.view(-1,28,28)
    images = images/2 + 0.5     # remove normalisation
    os.makedirs(test_img_folder, exist_ok=True)
    for img_idx in tqdm(range(len(images))):
        save_image(images[img_idx], test_img_folder+"/"+str(img_idx)+".png")
    # get and save summary statistics of test images
    compressed_filename = test_img_folder + ".npz"    
    save_stats_mnist(test_img_folder, compressed_filename) 


def test(vae, dl, dataset, latent_dim, nm_test=1024):
    # check if summary statistics of test dataset used for FID exist
    data_folder = './data'
    if not os.path.exists(data_folder + "/mnist_test.npz") :
        print(data_folder + "/mnist_test" + "does not exist")
        print("Creating compressed MNIST files for faster FID measure ...")
        make_compressed_MNIST_files(dataset, data_folder=data_folder)

    vae.eval()
    with torch.no_grad():
        z = torch.randn(nm_test, latent_dim).cuda()
        imgs = vae.decoder(z).cpu()

    img_folder = tmp_save_imgs(imgs.reshape(-1,28,28))
    # get inceptions score
    is_mean, is_std = get_mnist_inception_score(img_folder)

    # get mnist fid
    fid = fid_mnist(data_folder + "/mnist_test.npz", img_folder, device=torch.device("cpu"), num_workers=0, verbose=False)

    shutil.rmtree(img_folder)
    return is_mean, fid, imgs



def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    verbose = args.is_verbose    
    batch_size = args.batch_size
    lr = args.lr_p
    wd = args.decay_p
    activation = args.activation
    z_dim = 64
    h_dim = 256
    nm_epochs = 100
    
    if activation == "relu":
        activation = F.relu
    elif activation == 'tanh':
        activation = torch.tanh
    elif activation =='silu':
        activation = F.silu
    elif activation =='l-relu':
        activation = F.leaky_relu
    elif activation == "h-tanh":
        activation = F.hardtanh
    else:
        raise NotImplementedError

    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    # build model
    vae = VAE(x_dim=784, h_dim= h_dim, z_dim=z_dim, activation=activation)
    if torch.cuda.is_available():
        vae.cuda()

    optimizer = optim.Adam(vae.parameters(), lr = lr, weight_decay = wd)
    
    fids = []
    iss = []
    is_, fid, imgs = test(vae, test_loader, test_dataset, z_dim)
    if verbose:
        print(f"Epoch {0}/{nm_epochs} - Inception score: {is_ :.2f}, FID score: {fid :.2f}")
    for epoch in range(nm_epochs):
        train(epoch, vae, train_loader, optimizer, loss_function, verbose)
        if epoch % 10 == 9:
            is_, fid, imgs = test(vae, test_loader, test_dataset, z_dim)
            if verbose:
                print(f"Epoch {epoch + 1}/{nm_epochs} - Inception score: {is_ :.2f}, FID score: {fid :.2f}")
            fids.append(fid)
            iss.append(is_)
    is_, fid, imgs = test(vae, test_loader, test_dataset, z_dim)
    if verbose:
        print(f"Epoch {epoch+1}/{nm_epochs} - Inception score: {is_ :.2f}, FID score: {fid :.2f}")
    fids.append(fid)
    iss.append(is_)

    # imgs generated
    fig, axes = plt.subplots(10, 10, figsize=(10,10))
    images_reshaped = best_imgs.reshape(-1, 28, 28)
    axes = axes.ravel()
    for i in np.arange(0, 100):
        axes[i].imshow(images_reshaped[i], cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    plot_filename = f"plot_{uuid.uuid4().hex}.png"
    plt.savefig(plot_filename)
    if verbose:
       plt.show() 
    
    # fid and energy
    fid, axes = plt.subplots(1,2, figsize=(6,3))
    axes[0].plot(fids)
    axes[0].set_ylabel("fid")
    axes[1].plot(iss)
    axes[1].set_ylabel("inception score")
    plt.tight_layout()
    plot_filename = f"plot_{uuid.uuid4().hex}.png"
    plt.savefig(plot_filename)
    if verbose:
       plt.show() 

class Args:
    seed=None
    batch_size=None
    lr_p=None
    decay_p=None
    activation=None
    is_verbose=None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training a VAE model on MNIST.',
        fromfile_prefix_chars='@'
    )

    parser.add_argument('--seed', type=int, default=0,
        help='seed')
    parser.add_argument('--metric', type=str, default='fid', choices=['inception-score', 'fid'],
        help='metric for which the model has been optimised')    
    parser.add_argument('--is-verbose', type=lambda x: (str(x).lower() == 'true'), default=True,
        help='print outputs')    
    args = parser.parse_args()


    param = Args()
    param.seed = args.seed
    param.is_verbose = args.is_verbose

    if args.metric == "fid":
        param.batch_size = 150
        param.lr_p = 0.0005688033422898258
        param.decay_p = 0.01
        param.activation = "silu"
    elif args.metric == "inception-score":
        param.batch_size = 300
        param.lr_p = 0.0028592030010619043
        param.decay_p = 0.01
        param.activation = "silu"
    main(param)

