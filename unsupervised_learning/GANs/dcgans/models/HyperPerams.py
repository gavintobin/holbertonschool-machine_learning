#!/usr/bin/env python3
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch.functional as F
import pandas as  pd
import torchvision.transforms as T
from torchvision.utils import make_grid
import wandb
import time
wandb.init(project="your_project_name", name="your_experiment_name", save_code=True)


# preparing dataset using torchvison.datasets

transforms= T.Compose([
     T.ToTensor(),
     T.Normalize((0.5,), (0.5))
])

data_mnist= datasets.MNIST(root='.', train=True, transform=transforms, download=True)

batch_size =64
train_loader = DataLoader(dataset= data_mnist, batch_size =batch_size, shuffle= True)


# creating gen with linear layers, gan has 4 layers

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.flat =nn.Flatten()
    self.dis_model=nn.Sequential(
        #1st layer
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),

        #2nd Layer
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),

        #3rd Layer
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),

        #4th layer
        nn.Linear(256, 1),
        nn.Sigmoid()


    )

  def forward(self, x):

    x=self.flat(x)
    out=self.dis_model(x)

    return out

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
discriminate =Discriminator().to(device)


# vcreating gen model

class Generator(nn.Module):
  def __init__(self):
    super().__init__()

    self.gen_model=nn.Sequential(
        #1st layer
        nn.Linear(100, 256),
        nn.ReLU(),

        #2nd Layer
        nn.Linear(256, 512),
        nn.ReLU(),

        #3rd Layer
        nn.Linear(512, 1024),
        nn.ReLU(),

        #4th layer
        nn.Linear(1024, 784),
        nn.Tanh()
    )


  def forward(self, x):

    out=self.gen_model(x)
    out=out.view(x.size(0), 1, 28, 28)

    return out

generator =Generator().to(device)

#setting hyper perameters for model
lr=0.01
epochs=50
loss_function =nn.BCELoss()

optim_gen= torch.optim.SGD(generator.parameters(), lr=lr)
optim_dis= torch.optim.SGD(discriminate.parameters(), lr=lr)

# model training
# model training
for epoch in range(epochs):
    for n, (input_data, labels) in enumerate(train_loader):
        input_data = input_data.to(device)
        input_labels = torch.ones((input_data.size(0), 1)).to(device)  # Adjust size for real data batch
        noise = torch.randn((input_data.size(0), 100)).to(device)  # Adjust size for noise batch
        fake_labels = torch.zeros((input_data.size(0), 1)).to(device)  # Adjust size for fake labels
        generated_data = generator(noise)
        all_data = torch.cat((input_data, generated_data))
        all_labels = torch.cat((input_labels, fake_labels))

        # Training the discriminator
        discriminate.zero_grad()
        discriminate_output = discriminate(all_data)
        loss_discrminate = loss_function(discriminate_output, all_labels)
        loss_discrminate.backward()
        optim_dis.step()

        # Data for the generator
        noise = torch.randn((input_data.size(0), 100)).to(device)  # Adjust size for noise batch

        # Training the generator
        generator.zero_grad()
        generated_output = generator(noise)
        dis_gen_output = discriminate(generated_output)
        loss_generate = loss_function(dis_gen_output, input_labels)
        loss_generate.backward()
        optim_gen.step()

    #print loss
    # wandb.log({"loss_generate": loss_generate, "loss_discrminate": loss_discrminate, "Epoch ": epoch + 1, "Time ": time.time() - start})
    if n== batch_size-1:
      wandb.log({"loss_generate": loss_generate, "loss_discrminate": loss_discrminate, "Epoch ": epoch + 1, "Time ": time.time() - start})
      print(f'Epoch: {epoch+1} Loss Dis: {loss_discrminate}')
      print(f'Epoch: {epoch+1} Loss Gen: {loss_generate}')

# noise into model for testing after testing only gen is needeed for new images

noise= torch.randn((batch_size , 100)).to(device)
generated_data = generator(noise)

#lets hope this works
generated_data = generated_data.cpu().detach()
