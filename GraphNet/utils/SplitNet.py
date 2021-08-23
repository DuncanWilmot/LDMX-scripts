from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

from utils.ParticleNet import *

torch.set_default_dtype(torch.float32)

class SplitNet(nn.Module):
    # SplitNet class:
    # Consists of 3 ordinary ParticleNets, each one only examining data fom a single reigon of the ecal.
    # Hits from region 1 are fed to ParticleNet 1, hits from region 2 are fed to ParticleNet 2, etc.
    # The fully connected layer of each PN is replaced by a single fully connected layer that aggregates
    # the results of the one-to-three ParticleNets.
    # Number of regions is set by the nRegions param

    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128,0.1)],
                 use_fusion=False,
                 return_softmax=False,
                 nRegions=1,
                 **kwargs):
        super(SplitNet, self).__init__(**kwargs)
        print("INITIALIZING SPLITNET")

        self.nRegions = nRegions

        # Particle nets:  named pn1, pn2, etc.
        for i in range(self.nRegions):
            pn = ParticleNet(input_dims=input_dims, num_classes=2,           conv_params=conv_params,
                             fc_params=fc_params,   use_fusion=use_fusion,   return_softmax = return_softmax)
            setattr(self, 'pn{}'.format(i), pn)

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)

        # Fully connected layer:
        # NEW:  Modified PN to return x instead of output, moved FC layer here instead (and resized).
        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            fcs.append(nn.Sequential(nn.Linear(self.nRegions*in_chn, self.nRegions*channels), Mish(), nn.Dropout(drop_rate)))
        fcs.append(nn.Linear(self.nRegions*fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.return_softmax = return_softmax

        print("FINISHED INIT")

    """
    # No longer necessary after setattr() line above was added!
    def particle_nets_to(self, dev):  # Added separately--PNs in list aren't automatically put on gpu by to()
        return
        #for i in range(self.nRegions):
        #    setattr(self, 'pn{i}'.format(i), getattr(self, 'pn{i}'.format(i)).to(dev))
        #    self.particleNets[i] = self.particleNets[i].to(dev)
    """

    def forward(self, points, features):
        # Divide up provided points+features, then hand them to the PNs, then feed the outputs to the fc layer.
        # Points are [nregions] x 128  x 3 x 50 (note: nregions axis is gone for 1 region)
        # Note:  points[:,0].shape = (128, 3, 50)

        xi = [getattr(self, 'pn{}'.format(i))(points[:,i], features[:,i]) for i in range(self.nRegions)]

        output = self.fc(torch.cat(xi, dim=1))
        
        if self.return_softmax:
            output = torch.softmax(output, dim=1)
        return output


