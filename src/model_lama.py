# how much training is done in the model
# what kind of Discriminator is ued
# optimizer adam with 0.001 gen and 0.0001 for discriminator

import torch.nn as nn
import torch

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, img_width, single_side):
        super(Generator, self).__init__()

        

    def forward(self, x):
        x = self.model(x)

        return x


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(255),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=5, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, img_width, single_side, dropout):
        super(Discriminator, self).__init__()

        self.img_width = img_width
        self.single_side = single_side
        self.dropout = dropout

        self.global_disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=(9, 16), stride=1, padding=0, bias=False),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.local_disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=(9, 2), stride=1, padding=0, bias=False),
            nn.Dropout(self.dropout),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.concat_net = nn.Sequential(
            nn.Linear(1024*3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # get outputs from each of the discriminator networks
        global_output = self.global_disc(x)
        local_output_left = self.local_disc(x[:, :, :, :self.single_side])
        local_output_right = self.local_disc(x[:, :, :, (self.img_width - self.single_side):])

        # concatinates the outputs and sends them to the final layer
        # return local_output_right, global_output, local_output_left
        x = torch.cat((global_output.flatten(start_dim=1), local_output_left.flatten(start_dim=1), local_output_right.flatten(start_dim=1)), -1)
        x = self.concat_net(x)
        x = x.flatten()

        return x
