import torch.nn as nn
import torch
from snlayer import SpectralNorm
import numpy as np
import cv2

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock_G(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False, upsample=False, nobn=True):
        super(BasicBlock_G, self).__init__()
        self.upsample = upsample
        self.downsample = downsample
        self.nobn = nobn
        if self.upsample:
            self.conv1 = nn.ConvTranspose2d(inplanes, planes, 4, 2, 1)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        if not self.nobn:
            self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.downsample:
            self.conv2 =nn.Sequential(nn.AvgPool2d(2,2), conv3x3(planes, planes))
        else:
            self.conv2 = conv3x3(planes, planes)
        if not self.nobn:
            self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes or self.upsample or self.downsample:
            if self.upsample:
                self.skip = nn.ConvTranspose2d(inplanes, planes, 4, 2, 1)
            elif self.downsample:
                self.skip = nn.Sequential(nn.AvgPool2d(2,2), nn.Conv2d(inplanes, planes, 1, 1))
            else:
                self.skip = nn.Conv2d(inplanes, planes, 1, 1, 0)
        else:
            self.skip = None
        self.stride = stride

    def forward(self, x):
        residual = x
        if not self.nobn:
            out = self.bn1(x)
            out = self.relu(out)
        else:
            out = self.relu(x)
        out = self.conv1(out)
        if not self.nobn:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.skip is not None:
            residual = self.skip(x)
        out += residual
        return out

class BasicBlock_D(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False, nobn=True):
        super(BasicBlock_D, self).__init__()
        self.downsample = downsample
        self.nobn = nobn

        self.conv1 = SpectralNorm(nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        if not self.nobn:
            self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=False)
        if self.downsample:
            self.conv2 = nn.Sequential(nn.MaxPool2d(2, 2), SpectralNorm(
                nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)))
        else:
            self.conv2 = SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        if not self.nobn:
            self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes or self.downsample:
            if self.downsample:
                self.skip = nn.Sequential(nn.MaxPool2d(2, 2), SpectralNorm(nn.Conv2d(inplanes, planes, 1, 1)))
            else:
                self.skip = SpectralNorm(nn.Conv2d(inplanes, planes, 1, 1, 0))
        else:
            self.skip = None
        self.stride = stride

    def forward(self, x):
        residual = x
        if not self.nobn:
            out = self.bn1(x)
            out = self.relu(out)
        else:
            out = self.relu(x)
        out = self.conv1(out)
        if not self.nobn:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.skip is not None:
            residual = self.skip(x)
        out += residual
        return out

class G_DHL(nn.Module):
    def __init__(self):
        super(G_DHL, self).__init__()
        blocks = [96, 96, 128, 128, 256, 256, 512, 512, 128, 128, 32, 32]

        self.noise_fc = nn.Linear(64, 4096)
        self.in_layer = nn.Conv2d(4, blocks[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.out_layer = nn.Sequential(
            nn.Conv2d(blocks[-1], 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        downs = []
        in_feat = blocks[0]
        for i in range(8):
            b_down = not i % 2
            downs.append(BasicBlock_G(in_feat, blocks[i], downsample=b_down))
            in_feat = blocks[i]

        ups = []
        for i in range(2):
            ups.append(nn.PixelShuffle(2))
            ups.append(BasicBlock_G(blocks[8+i*2], blocks[8+i*2]))
            ups.append(BasicBlock_G(blocks[9+i*2], blocks[9+i*2]))

        self.down_layers = nn.Sequential(*downs)
        self.up_layers = nn.Sequential(*ups)

    def forward(self, x, z):
        noises = self.noise_fc(z)
        noises = noises.view(-1, 1, 64, 64)
        out = torch.cat((x, noises), 1)
        out = self.in_layer(out)
        out = self.down_layers(out)
        out = self.up_layers(out)
        out = self.out_layer(out)
        return out

class G_RLS(nn.Module):
    def __init__(self, ngpu=1):
        super(G_RLS, self).__init__()
        self.ngpu = ngpu
        res_units = [256, 128, 96]
        inp_res_units = [
            [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
             256], [256, 128, 128], [128, 96, 96]]

        self.layers_set = []
        self.layers_set_up = []
        self.layers_set_final = nn.ModuleList()
        self.layers_set_final_up = nn.ModuleList()

        self.a1 = nn.Sequential(nn.Conv2d(256, 128, 1, 1))
        self.a2 = nn.Sequential(nn.Conv2d(128, 96, 1, 1))

        self.layers_in = conv3x3(3, 256)

        layers = []
        for ru in range(len(res_units) - 1):
            nunits = res_units[ru]
            curr_inp_resu = inp_res_units[ru]
            self.layers_set.insert(ru, [])
            self.layers_set_up.insert(ru, [])

            if ru == 0:
                num_blocks_level = 12
            else:
                num_blocks_level = 3

            for j in range(num_blocks_level):
                self.layers_set[ru].append(BasicBlock_G(curr_inp_resu[j], nunits))

            self.layers_set_up[ru].append(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True))

            self.layers_set_up[ru].append(nn.BatchNorm2d(nunits))
            self.layers_set_up[ru].append(nn.ReLU(True))
            self.layers_set_up[ru].append(nn.ConvTranspose2d(nunits, nunits, kernel_size=1, stride=1))
            self.layers_set_final.append(nn.Sequential(*self.layers_set[ru]))
            self.layers_set_final_up.append(nn.Sequential(*self.layers_set_up[ru]))

        nunits = res_units[-1]
        layers.append(conv3x3(inp_res_units[-1][0], nunits))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(inp_res_units[-1][1], nunits, kernel_size=1, stride=1))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(nunits, 3, kernel_size=1, stride=1))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers_in(input)
        for ru in range(len(self.layers_set_final)):
            if ru == 0:
                temp = self.layers_set_final[ru](x)
                x = x + temp
            elif ru == 1:
                temp = self.layers_set_final[ru](x)
                temp2 = self.a1(x)
                x = temp + temp2
            elif ru == 2:
                temp = self.layers_set_final[ru](x)
                temp2 = self.a2(x)
                x = temp + temp2
            x = self.layers_set_final_up[ru](x)

        x = self.main(x)

        return x

class G_DSL(nn.Module):
    def __init__(self):
        super(G_DSL, self).__init__()
        blocks = [96, 96, 128, 128, 256, 256, 512, 512, 128, 128, 32, 32]

        self.noise_fc = nn.Linear(64, 4096)
        self.in_layer = nn.Conv2d(4, blocks[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.out_layer = nn.Sequential(
            nn.Conv2d(blocks[-1], 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        downs = []
        in_feat = blocks[0]
        for i in range(8): # downsample layers
            b_down = not i % 2
            downs.append(BasicBlock_G(in_feat, blocks[i], downsample=b_down))
            in_feat = blocks[i]

        ups = []
        for i in range(2):
            ups.append(nn.PixelShuffle(2))
            ups.append(BasicBlock_G(blocks[8+i*2], blocks[8+i*2]))
            ups.append(BasicBlock_G(blocks[9+i*2], blocks[9+i*2]))

        self.down_layers = nn.Sequential(*downs)
        self.up_layers = nn.Sequential(*ups)

    def forward(self, x, z):
        noises = self.noise_fc(z)
        noises = noises.view(-1, 1, 64, 64)
        out = torch.cat((x, noises), 1)
        out = self.in_layer(out)
        out = self.down_layers(out)
        out = self.up_layers(out)
        out = self.out_layer(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        assert input_size in [16, 64]
        self.blocks = [128, 128, 256, 256, 512, 512]
        pool_start = len(self.blocks) - 4 if input_size == 64 else len(self.blocks) - 2
        self.out_layer = nn.Sequential(
            SpectralNorm(nn.Linear(16*self.blocks[-1], self.blocks[-1])),
            nn.ReLU(),
            SpectralNorm(nn.Linear(self.blocks[-1], 1))
        )

        rbs = []
        in_feat = 3
        for i in range(len(self.blocks)):
            b_down = bool(i >= pool_start)
            rbs.append(BasicBlock_D(in_feat, self.blocks[i], downsample=b_down, nobn=True))
            in_feat = self.blocks[i]
        self.residual_blocks = nn.Sequential(*rbs)

    def forward(self, x):
        out = self.residual_blocks(x)
        out = out.view(-1, 16*self.blocks[-1])
        out = self.out_layer(out)
        return out

def discriminator_test():
    in_size = 64
    net = Discriminator(in_size).cuda()
    X = np.random.randn(2, 3, in_size, in_size).astype(np.float32)
    X = torch.from_numpy(X).cuda()
    Y = net(X)
    print(Y.shape)

def G_DHL_test():
    net = G_DHL().cuda()
    Z = np.random.randn(1, 1, 64).astype(np.float32)
    X = np.random.randn(1, 3, 64, 64).astype(np.float32)
    Z = torch.from_numpy(Z).cuda()
    X = torch.from_numpy(X).cuda()
    Y = net(X, Z)
    print(Y.shape)
    Xim = X.cpu().numpy().squeeze().transpose(1, 2, 0)
    Yim = Y.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    Xim = (Xim - Xim.min()) / (Xim.max() - Xim.min())
    Yim = (Yim - Yim.min()) / (Yim.max() - Yim.min())
    cv2.imshow("X", Xim)
    cv2.imshow("Y", Yim)
    cv2.waitKey()

if __name__ == "__main__":
    net = G_RLS().cuda()
    X = np.random.randn(1, 3, 16, 16).astype(np.float32) # B, C, H, W
    X = torch.from_numpy(X).cuda()
    Y = net(X)
    print(Y.shape)
    Xim = X.cpu().numpy().squeeze().transpose(1,2,0)
    Yim = Y.detach().cpu().numpy().squeeze().transpose(1,2,0)
    Xim = (Xim - Xim.min()) / (Xim.max() - Xim.min())
    Yim = (Yim - Yim.min()) / (Yim.max() - Yim.min())
    cv2.imshow("X", Xim)
    cv2.imshow("Y", Yim)
    cv2.waitKey()
    print("finished.")
    G_DHL_test()
    discriminator_test()
    print("finished.")
