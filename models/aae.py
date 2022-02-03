import torch
import torch.nn as nn

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.z_h = config['latent_image_height']
        self.z_w = config['latent_image_width']
        self.use_bias = config['model']['G']['use_bias']
        self.relu_slope = config['model']['G']['relu_slope']
        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=2048 * 3, bias=self.use_bias),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.z_h*self.z_w*3, self.z_size, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        flattened_input = torch.flatten(input, start_dim=1)
        z=self.fc2(flattened_input)
        output = self.model(z)
        output = output.view(-1, 3, 2048)
        return output


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.z_h = config['latent_image_height']
        self.z_w = config['latent_image_width']
        self.use_bias = config['model']['D']['use_bias']
        self.relu_slope = config['model']['D']['relu_slope']
        self.dropout = config['model']['D']['dropout']

        self.model = nn.Sequential(

            nn.Linear(self.z_size, 512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(512, 128, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(64, 1, bias=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.z_h*self.z_w*3, self.z_size, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        flattened_input = torch.flatten(x, start_dim=1)
        z=self.fc2(flattened_input)
        logit = self.model(z)
        return logit


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.z_h = config['latent_image_height']
        self.z_w = config['latent_image_width']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
        )

        # self.m2i = nn.Sequential(nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        #                         nn.Sigmoid())

        # self.fc = nn.Sequential(
        #     nn.Linear(256, 256, bias=True),
        #     nn.ReLU(inplace=True)
        # )

        # self.mu_layer = nn.Linear(256, self.z_size, bias=True)
        # self.std_layer = nn.Linear(256, self.z_size, bias=True)
        self.decoder_input = nn.Sequential(nn.Linear(in_features=256, out_features=self.z_h*self.z_w*3),
                            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.conv(x)
        # mat1 = output.view(-1,32,128,128)
        # mat2= self.m2i(mat1)
        # vect=torch.flatten(output, start_dim=1)
        output2 = output.max(dim=2)[0]
        # logit = self.fc(output2)
        # mu = self.mu_layer(logit)
        # logvar = self.std_layer(logit)
        # z = self.reparameterize(mu, logvar)
        image_format = self.decoder_input(output2)
        image_format = image_format.view(-1, 3, self.z_h, self.z_w)
        return image_format
