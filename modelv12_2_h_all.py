import torch.nn as nn
from DWT_IDWT.DWT_IDWT_layer import *
import torch
from quality_assessment import calc_cc_tensor
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import numpy as np

import math


# when kernel_size=3/5/7/9/11/13,stride=1,dilation=1, accordingly padding=1/2/4/5/6  the w and h of images stay the same
# MAIN PART CC+HIGH FREQUENCY-CB+CC LOSS
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups)  # , bias=False


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, groups=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2)  # nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)

        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# data and data similar
class FC_mine(nn.Module):
    # MLP to change high dimension feature to low dimension feature for CC loss calculation
    def __init__(self, dv, dv1, do):
        # dv is the M*N*C number of input feature-(B C M N)
        super(FC_mine, self).__init__()

        self.relu = nn.LeakyReLU(0.2)
        num_hid = int(math.sqrt(dv))

        self.bn0 = nn.BatchNorm1d(num_hid, affine=False)

        self.layer1 = nn.Sequential(nn.Linear(dv, num_hid), self.bn0, self.relu, nn.Linear(num_hid, do))
        self.layer2 = nn.Sequential(nn.Linear(dv1, num_hid), self.bn0, self.relu, nn.Linear(num_hid, do))
        self.bn = nn.BatchNorm1d(do, affine=False)

    def forward(self, F1, F2):
        # change the shape from (B C M N) to (B M*N*C) for nn.linear function
        F1 = torch.reshape(F1, (F1.size(0), F1.size(1) * F1.size(2) * F1.size(3)))
        F2 = torch.reshape(F2, (F2.size(0), F2.size(1) * F2.size(2) * F2.size(3)))
        # print("F1.shape:", F1.shape, F2.shape)

        # reduce dimension from (B M*N*C) to (B M*N*C)
        F1_1 = self.bn(self.layer1(F1))
        F2_1 = self.bn(self.layer2(F2))

        return F1_1, F2_1


# data and data similar
class FC_mine1(nn.Module):
    # MLP to change high dimension feature to low dimension feature for CC loss calculation
    def __init__(self, dv, do):
        # dv is the M*N*C number of input feature-(B C M N)
        super(FC_mine1, self).__init__()

        self.relu = nn.LeakyReLU(0.2)
        num_hid = int(math.sqrt(dv))  # 64 #

        do = 4 * int(math.sqrt(num_hid))  # int(math.sqrt(num_hid))
        print('do:', do)

        self.bn0 = nn.BatchNorm1d(num_hid, affine=False)

        self.layer1 = nn.Sequential(nn.Linear(dv, num_hid), self.bn0, self.relu, nn.Linear(num_hid, do))
        self.layer2 = nn.Sequential(nn.Linear(dv, num_hid), self.bn0, self.relu, nn.Linear(num_hid, do))
        self.bn = nn.BatchNorm1d(do, affine=False)

    def forward(self, F1, F2):
        # change the shape from (B C M N) to (B M*N*C) for nn.linear function
        F1 = torch.reshape(F1, (F1.size(0), F1.size(1) * F1.size(2) * F1.size(3)))
        F2 = torch.reshape(F2, (F2.size(0), F2.size(1) * F2.size(2) * F2.size(3)))
        # print("F1.shape:", F1.shape, F2.shape)

        # reduce dimension from (B M*N*C) to (B M*N*C)
        F1_1 = self.bn(self.layer1(F1))
        F2_1 = self.bn(self.layer2(F2))
        return F1_1, F2_1



class WavePNet(nn.Module):
    def __init__(self, wavename='haar', bands=4, c1=48, c2=48, c11=48, c21=48, dv=12288, do=32):# do=64
        # c1=64, c2=64, c11=64, c21=64, c3=64, c31=64    c1=32, c2=32, c11=32, c21=32, c3=32, c31=32
        # 16 * 16* 32 =8192
        super(WavePNet, self).__init__()

        self.relu = nn.LeakyReLU(0.2)

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            self.relu,
            nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            self.relu
        )  # self._make_layer(BasicBlock, 1, c1)
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=c11 * 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c11 * 3),
            self.relu,
            nn.Conv2d(in_channels=c11 * 3, out_channels=c11 * 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c11 * 3),
            self.relu
        )  # self._make_layer(BasicBlock, 3, c11 * 3)
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=bands, out_channels=c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            self.relu,
            nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c1),
            self.relu,
            # ChannelAttention(c1) AttentionBlock(c1)
        )  # self._make_layer(BasicBlock, bands, c1)
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=bands * 3, out_channels=c11 * 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c11 * 3),
            self.relu,
            nn.Conv2d(in_channels=c11 * 3, out_channels=c11 * 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c11 * 3),
            self.relu
        )  # self._make_layer(BasicBlock, bands * 3, c11 * 3)

        self.conv2_1 = self._make_layer(BasicBlock, c1, c2)

        self.conv2_2 = self._make_layer(BasicBlock, c1 * 3, c21 * 3)

        self.conv2_3 = self._make_layer(BasicBlock, c1, c2)

        self.conv2_4 = self._make_layer(BasicBlock, c1 * 3, c21 * 3)

        # # ___________________________________________add___________________________________________
        self.conv3 = self._make_layer(BasicBlock, c2, c21)
        self.conv3_h = self._make_layer(BasicBlock, c21 * 3, c21 * 3)

        self.conv4 = self._make_layer(BasicBlock, c21, c11)
        self.conv4_h = self._make_layer(BasicBlock, c11 * 3, c11 * 3)

        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=c11, out_channels=bands, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(bands),
                                   self.relu,  # self.relu
                                   nn.Conv2d(in_channels=bands, out_channels=bands, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(bands),
                                   nn.Tanh()
                                   )

        self.dwt0 = DWT_2D(wavename=wavename)
        self.after_dwt0 = nn.BatchNorm2d(1)
        self.dwt0_1 = DWT_2D(wavename=wavename)
        self.after_dwt0_1 = nn.BatchNorm2d(bands)
        self.dwt1 = DWT_2D(wavename=wavename)
        self.after_dwt1 = nn.BatchNorm2d(c1)
        self.dwt1_1 = DWT_2D(wavename=wavename)
        self.after_dwt1_1 = nn.BatchNorm2d(c1)

        # ___________________________________________________add_______________________________________________________
        self.idwt0 = IDWT_2D(wavename=wavename)
        self.after_idwt0 = nn.BatchNorm2d(c11)
        self.idwt1 = IDWT_2D(wavename=wavename)
        self.after_idwt1 = nn.BatchNorm2d(c21)

        # ___________________________________________________HFS  loss_______________________________________________________
        # self.latent_lh = FC_mine1(3*dv, do)
        # self.latent_h = FC_mine1(3*4 * dv, do)
        # ___________________________________________________fusion part_______________________________________________________
        self.fusion_h2 = nn.Conv2d(in_channels=2 * c21 * 3, out_channels=c21 * 3, kernel_size=1, stride=1, padding=0)
        self.fusion_h1 = nn.Conv2d(in_channels=2 * c11 * 3, out_channels=c11 * 3, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, inplanes, out_planes):
        layers = []
        layers.append(block(inplanes, out_planes))

        return nn.Sequential(*layers)

    def _make_layer2(self, block, inplanes, out_planes, block2):
        layers = []
        layers.append(block(inplanes, out_planes))
        layers.append(block2(out_planes))
        return nn.Sequential(*layers)

    def forward(self, pan, ms):
        """ pan road """

        pd_0, pg_01, pg_02, pg_03 = self.dwt0(pan)  # LL, LH, HL, HH  def forward(self, input):
        pd_0 = self.after_dwt0(pd_0)
        pg_01 = self.after_dwt0(pg_01)
        pg_02 = self.after_dwt0(pg_02)
        pg_03 = self.after_dwt0(pg_03)

        # print("2")  # convert to tensor
        pg_0 = torch.cat((pg_01, pg_02, pg_03), 1)
        pc_1 = self.conv1_1(pd_0)
        pgc_1 = self.conv1_2(pg_0)

        pd_1, pg_11, pg_12, pg_13 = self.dwt1(pc_1)
        pd_1 = self.after_dwt1(pd_1)
        pg_11 = self.after_dwt1(pg_11)
        pg_12 = self.after_dwt1(pg_12)
        pg_13 = self.after_dwt1(pg_13)

        pg_1 = torch.cat((pg_11, pg_12, pg_13), 1)
        pc_2 = self.conv2_1(pd_1)
        pgc_2 = self.conv2_2(pg_1)

        '''ms road'''

        md_0, mg_01, mg_02, mg_03 = self.dwt0_1(ms)  # LL, LH, HL, HH
        md_0 = self.after_dwt0_1(md_0)
        mg_01 = self.after_dwt0_1(mg_01)
        mg_02 = self.after_dwt0_1(mg_02)
        mg_03 = self.after_dwt0_1(mg_03)

        mg_0 = torch.cat((mg_01, mg_02, mg_03), 1)
        mc_1 = self.conv1_3(md_0)
        mgc_1 = self.conv1_4(mg_0)

        md_1, mg_11, mg_12, mg_13 = self.dwt1_1(mc_1)
        md_1 = self.after_dwt1_1(md_1)
        mg_11 = self.after_dwt1_1(mg_11)
        mg_12 = self.after_dwt1_1(mg_12)
        mg_13 = self.after_dwt1_1(mg_13)

        mg_1 = torch.cat((mg_11, mg_12, mg_13), 1)
        mc_2 = self.conv2_3(md_1)  # CC2
        mgc_2 = self.conv2_4(mg_1)

        low = mc_2
        flc_3 = self.conv3(low)

        # high2 = pgc_2
        high2 = self.fusion_h2(torch.cat([pgc_2, mgc_2], dim=1))
        high2 = self.conv3_h(high2)

        _, channel2, _, _ = high2.shape

        c2 = int(channel2 / 3)  # 16.0 without int()
        fli_1 = self.idwt1(flc_3, high2[:, 0:c2, :, :], high2[:, c2:2 * c2, :, :],
                           high2[:, 2 * c2:channel2, :, :])  # def forward(self, LL, LH, HL, HH):
        fli_1 = self.after_idwt1(fli_1)

        flc_4 = self.conv4(fli_1)

        # high1 = pgc_1
        high1 = self.fusion_h1(torch.cat([pgc_1, mgc_1], dim=1))
        high1 = self.conv4_h(high1)

        _, channel1, _, _ = high1.shape
        c1 = int(channel1 / 3)
        fli_0 = self.idwt0(flc_4, high1[:, 0:c1, :, :], high1[:, c1:2 * c1, :, :],
                           high1[:, 2 * c1:channel1, :, :])
        fli_0 = self.after_idwt0(fli_0)

        fused = self.conv5(fli_0)

        # ___________________________________________________HFS  loss_______________________________________________________
        # panlh_latent, mslh_latent = self.latent_lh(pgc_2, mgc_2)
        # panh_latent, msh_latent = self.latent_h(pgc_1, mgc_1)
        # when you add HFS  loss, remenber to return "panh_latent, msh_latent, panlh_latent, mslh_latent"
       
        return fused, pc_1, pgc_1, pc_2, pgc_2, mc_1, mgc_1, mc_2, mgc_2, low, high2, high1#, panh_latent, msh_latent, panlh_latent, mslh_latent 
