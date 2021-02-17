import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pybulletX.utils.space_dict import SpaceDict
from utils.network import calc_out_size

def is_tactile_in_obs(observation_space):
    if isinstance(observation_space, SpaceDict):
        keys = list(observation_space.keys())
        if "tactile_sensor" in keys:
            return True
    return False

def get_encoder_network(observation_space):
    h, w = observation_space[0].shape
    h, w = calc_out_size(h, w, 8, stride=4)
    h, w = calc_out_size(h, w, 4, stride=2)
    return nn.Sequential(OrderedDict([
            ('enc_cnn_1', nn.Conv2d(1, 16, 8, stride=4)),
            ('enc_cnn_elu_1', nn.ELU()),
            ('enc_cnn_2', nn.Conv2d(16, 32, 4, stride=2)),
            ('spatial_softmax', SpatialSoftmax(h, w)), #Batch_size, 2 * num_channels
        ]))

class TactileNetwork(nn.Module):
    def __init__(self, encoder_network, output_dim):
        super(TactileNetwork, self).__init__()
        self.encoder_network = encoder_network
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x, detach_encoder):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        s1 = self.encoder_network(x[:,0].unsqueeze(1)).squeeze()
        s2 = self.encoder_network(x[:,1].unsqueeze(1)).squeeze()
        output = torch.cat((s1, s2), dim=-1)
        if detach_encoder:
            output = output.detach()
        output = self.fc(output)
        return output

class SpatialSoftmax(nn.Module):
    # reference: https://arxiv.org/pdf/1509.06113.pdf
    def __init__(self, height, width):
        super(SpatialSoftmax, self).__init__()
        x_map = np.empty([height, width], np.float32)
        y_map = np.empty([height, width], np.float32)

        for i in range(height):
            for j in range(width):
                x_map[i, j] = (i - height / 2.0) / height
                y_map[i, j] = (j - width / 2.0) / width

        self.x_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).cuda() # W*H
        self.y_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).cuda() # W*H

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]) # batch, C, W*H
        x = F.softmax(x, dim=2) # batch, C, W*H
        fp_x = torch.matmul(x, self.x_map) # batch, C
        fp_y = torch.matmul(x, self.y_map) # batch, C
        x = torch.cat((fp_x, fp_y), 1)
        return x # batch, C*2class SpatialSoftmax(nn.Module):
    # reference: https://arxiv.org/pdf/1509.06113.pdf
    def __init__(self, height, width):
        super(SpatialSoftmax, self).__init__()
        x_map = np.empty([height, width], np.float32)
        y_map = np.empty([height, width], np.float32)

        for i in range(height):
            for j in range(width):
                x_map[i, j] = (i - height / 2.0) / height
                y_map[i, j] = (j - width / 2.0) / width

        self.x_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).cuda() # W*H
        self.y_map = torch.from_numpy(np.array(x_map.reshape((-1)), np.float32)).cuda() # W*H

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]) # batch, C, W*H
        x = F.softmax(x, dim=2) # batch, C, W*H
        fp_x = torch.matmul(x, self.x_map) # batch, C
        fp_y = torch.matmul(x, self.y_map) # batch, C
        x = torch.cat((fp_x, fp_y), 1)
        return x # batch, C*2

class DecoderNetwork(nn.Module):
    def __init__(self, observation_space, tactile_dim):
        super(DecoderNetwork, self).__init__()
        self.i_h, self.i_w = observation_space[0].shape
        h, w = self.i_h, self.i_w
        h, w = calc_out_size(h, w, 8, stride=4)
        self.h, self.w = calc_out_size(h, w, 4, stride=2)
        self.fc = nn.Linear(tactile_dim, 128)
        self.fc2 = nn.Linear(64, self.h * self.w * 32) # Instead of Spatial Softmax
        self.decoder = nn.Sequential(OrderedDict([
                        ('dec_cnn_trans_1', nn.ConvTranspose2d(32, 16, 4, stride=2)),
                        ('dec_cnn_trans_elu_1', nn.ELU()),
                        ('dec_cnn_trans_2', nn.ConvTranspose2d(16, 1, 8, stride=4))
                    ]))

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.elu(self.fc(x))
        # Image tactile left
        s1 = F.elu(self.fc2(x[:,:64]))
        s1 = s1.view(batch_size, 32, self.h, self.w)
        s1 = self.decoder(s1) 
        # Image tactile right
        s2 = F.elu(self.fc2(x[:,64:]))
        s2 = s2.view(batch_size, 32, self.h, self.w)
        s2 = self.decoder(s2) 
        output = torch.cat((s1,s2), dim=1)
        output = F.interpolate(output, size=(self.i_h, self.i_w))
        return output