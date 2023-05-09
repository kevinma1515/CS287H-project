import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from scipy.stats import norm

from utility import get_celeba
from dcgan import weights_init, Generator, Discriminator
import PySimpleGUI as sg
from PIL import Image

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

# Set random seed for reproducibility.
seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Parameters to define the model.
params = {
    "bsize": 128,  # Batch size during training.
    'imsize': 64,  # Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc': 3,  # Number of channles in the training images. For coloured images this is 3.
    'nz': 100,  # Size of the Z latent vector (the input to the generator).
    'ngf': 64,  # Size of feature maps in the generator. The depth will be multiples of this.
    'ndf': 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs': 10,  # Number of training epochs.
    'lr': 0.0002,  # Learning rate for optimizers
    'beta1': 0.5,  # Beta1 hyperparam for Adam optimizer
    'save_epoch': 2}  # Save step.

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

# Get the data.
dataloader = get_celeba(params)

# Plot the training images.
sample_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[: 64], padding=2, normalize=True).cpu(), (1, 2, 0)))

plt.show()

# Create the generator.
netG = Generator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netG.apply(weights_init)
# Print the model.
print(netG)

# Create the discriminator.
netD = Discriminator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netD.apply(weights_init)
# Print the model.
print(netD)

# Binary Cross Entropy loss function.
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

real_label = 1
fake_label = 0

# Optimizer for the discriminator.
optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
# Optimizer for the generator.
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

# Stores generated images as training progresses.
img_list = []
# Stores generator losses during training.
G_losses = []
# Stores discriminator losses during training.
D_losses = []

iters = 0

print("Starting Training Loop...")
print("-" * 25)

# img1 and img2 should be a file directory to your png file
def gui_pref(img1, img2):
    layout = [[sg.Text("Which image do you prefer?")],
              [sg.Image(img1)], [sg.Image(img2)],
              [sg.Button('1st Image')], [sg.Button('2nd Image')], [sg.Button('Neither')]]

    # Create the window
    window = sg.Window('Window Title', layout, resizable = True)

    # Display and interact with the window using an event loop
    while True:
        event, values = window.read()
        # see if the user wants to quit or window was closed
        if event == '1st Image':
            l_1 = 1
            l_2 = 0
            break
        elif event == '2nd Image':
            l_1 = 0
            l_2 = 1
            break
        elif event == 'Neither':
            l_1 = 0.5
            l_2 = 0.5
            break
        elif event == sg.WIN_CLOSED:
            break

    window.close()
    return (l_1, l_2)

def pref_loss(l_1, l_2, r_1, r_2):
    f1_2 = torch.exp(torch.sum(r_1))/(torch.exp(torch.sum(r_2))+torch.exp(torch.sum(r_1)))
    f2_1 = torch.exp(torch.sum(r_2))/(torch.exp(torch.sum(r_2))+torch.exp(torch.sum(r_1)))

    return (-(l_1*torch.log(f1_2)+l_2*torch.log(f2_1)), f1_2, f2_1)

class LatentNetwork(nn.Module):
    def __init__(self, params):
        super().__init__()
        # Input is the latent vector z
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(72*params['nz'], 72*params['nz']),
            nn.ReLU(),
            nn.Linear(72*params['nz'], 72*params['nz']),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)

        return logits

pref_model = LatentNetwork(params).to(device)
optimizerPref = optim.Adam(pref_model.parameters())

f1_2 = []
f2_1 = []
# first train generator and discriminator
for epoch in range(params['nepochs']):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
    if epoch % 2 == 0:
        print(' it is moving?')
# second do human preference modelling
for epoch in range(params['nepochs']):
    for i, data in enumerate(dataloader, 0):


        ## Train with all-real batch

        # Transfer data tensor to GPU/CPU (device)
        real_data = data[0].to(device)
        # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
        b_size = 72
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        #############################################################################


        #############################################################################
        # (2) Train latent reward network
        #############################################################################

        # sample (1) generated from a normal distribution
        # size b_size x params['nz'] x 1 x 1
        noise_1 = torch.randn(b_size, params['nz'], 1, 1, device = device)

        # sample (2) generated from a normal distribution
        noise_2 = torch.randn(b_size, params['nz'], 1, 1, device = device)

        # go through the network by picking human preferences first
        fake_data_1 = netG(noise_1)
        fake_data_2 = netG(noise_2)
        # we will need to implement a way to have l_1 and l_2 generated based on human preferences from fake_data_1 and fake_data_2
        # for now I just do a random test
        if i % 500 == 0:
            with torch.no_grad():
                fake_data_r1 = netG(noise_1).detach().cpu()
                test_img1 = vutils.make_grid(fake_data_r1, padding=2, normalize=True)
                fake_data_r2 = netG(noise_2).detach().cpu()
                test_img2 = vutils.make_grid(fake_data_r2, padding=2, normalize=True)
            plt.imshow(np.transpose(test_img1, (1, 2, 0)))
            plt.savefig('Data/png/img1.png')
            plt.imshow(np.transpose(test_img2, (1, 2, 0)))
            plt.savefig('Data/png/img2.png')
            pref_values = gui_pref('Data/png/img1.png', 'Data/png/img2.png')
            l_1 = pref_values[0]
            l_2 = pref_values[1]

        optimizerPref.zero_grad()
        netG.zero_grad()
        loglik1 = pref_model(noise_1.flatten())
        loglik2 = pref_model(noise_2.flatten())
        errPref = pref_loss(l_1, l_2, loglik1, loglik2)
        errPref[0].backward()
        optimizerG.step()
        optimizerPref.step()


        f1_2.append(errPref[1])
        f2_1.append(errPref[2])

    if epoch % params['save_epoch'] == 0:
        print("it is working, I guess")


# ## Try Negative Sampling some other time
# def p(x):
#     return pref_model(x)
#
# def q(x):
#     return norm.pdf(x = x, loc = 0, scale = 1)
#
# noise = torch.randn(b_size, params['nz'], 1, 1, device = device)
# p_max = torch.max(pref_model(noise.flatten()))
#
# def sample():
#     xs = np.random.normal(0, 1, size = 1800)
#     cs = np.random.uniform(0, 1, size = 1800)
#
#     cs = torch.rand(size = 1800)
#     return xs[mask]

def error_latent(latent):
    return -torch.sum(latent)

# retrain generator and discriminator based on human preferences
for epoch in range(params['nepochs']):
    for i, data in enumerate(dataloader, 0):
        #############################################################################
        # (3) Train GAN using latent reward network
        #############################################################################

        # Transfer data tensor to GPU/CPU (device)
        real_data = data[0].to(device)
        # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
        b_size = real_data.size(0)
        # Make accumulated gradients of the discriminator zero.
        netD.zero_grad()
        # Create labels for the real data (label = 1)
        label = torch.full((b_size, ), real_label, device=device)
        label = label.float()
        output = netD(real_data).view(-1)
        errD_real = criterion(output, label)
        # Calculate gradients for backpropogation
        errD_real.backward()
        D_x = output.mean().item()

        b_size = 72
        label = torch.full((b_size, ), real_label, device=device)
        label = label.float()
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        # Update G network: Maximize log(D(G(z))
        netG.zero_grad()
        label.fill_(real_label) # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        D_G_z2 = output.mean().item()

        # Update total loss: Latent Loss + log(D(G(z))
        # do a forward pass in latent network
        optimizerPref.zero_grad()
        loglik = pref_model(noise.flatten())
        errL = error_latent(loglik)
        err_total = errG + errL
        err_total.backward()
        # Update G
        optimizerG.step()
        # update latent network
        optimizerPref.step()

        # Save the losses for plotting.
        # uncomment these for later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on a fixed noise.
        if (iters % 100 == 0) or ((epoch == params['nepochs'] - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake_data = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))

        iters += 1

    # Save the model.
    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'params': params
        }, 'model/model_epoch_{}.pth'.format(epoch))

# Save the final trained model.
torch.save({
    'generator': netG.state_dict(),
    'discriminator': netD.state_dict(),
    'optimizerG': optimizerG.state_dict(),
    'optimizerD': optimizerD.state_dict(),
    'params': params
}, 'model/model_final.pth')

# Plot the training losses.
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Animation showing the improvements of the generator.
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
anim.save('celeba.gif', dpi=80, writer='imagemagick')