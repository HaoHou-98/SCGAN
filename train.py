import os, sys
import numpy as np
import cv2
import random
import torch
import time
import torch.nn as nn
import torch.optim as optim
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from data_train import faces_data, High_Data, Low_Data
from data_test import get_loader
from model import G_DHL, G_RLS, G_DSL, Discriminator
from torch.autograd import Variable
from torch.optim import lr_scheduler
import argparse
import torch.nn.functional as F
from tqdm import tqdm,trange
from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser()

writer = SummaryWriter('checkpoints')


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    opt = edict()
    opt.batchsize = 1
    seed_num = 2020
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    max_epoch = 200
    learn_rate = 1e-4
    alpha, beta = 1, 0.05
    theta, gamma = 1, 0.05

    G_DHL = G_DHL().cuda()
    G_RLS = G_RLS().cuda()
    G_DSL = G_DSL().cuda()
    D_L1 = Discriminator(16).cuda()
    D_L2 = Discriminator(16).cuda()
    D_H1 = Discriminator(64).cuda()
    D_H2 = Discriminator(64).cuda()

    L1Loss = nn.L1Loss(reduction='mean')

    optim_G_DHL = optim.Adam(G_DHL.parameters(), lr=learn_rate, betas=(0.9, 0.999))
    optim_G_RLS = optim.Adam(G_RLS.parameters(), lr=learn_rate, betas=(0.9, 0.999))
    optim_G_DSL = optim.Adam(G_DSL.parameters(), lr=learn_rate, betas=(0.9, 0.999))
    optim_D_L1 = optim.Adam(D_L1.parameters(), lr=learn_rate, betas=(0.9, 0.999))
    optim_D_L2 = optim.Adam(D_L2.parameters(), lr=learn_rate, betas=(0.9, 0.999))
    optim_D_H1 = optim.Adam(D_H1.parameters(), lr=learn_rate, betas=(0.9, 0.999))
    optim_D_H2 = optim.Adam(D_H2.parameters(), lr=learn_rate, betas=(0.9, 0.999))

    CosineLR1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_G_DHL, T_max=10, eta_min=0.00001)
    CosineLR2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_G_RLS, T_max=10, eta_min=0.00001)
    CosineLR3 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_G_DSL, T_max=10, eta_min=0.00001)
    CosineLR4 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_D_L1, T_max=10, eta_min=0.00001)
    CosineLR5 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_D_L2, T_max=10, eta_min=0.00001)
    CosineLR6 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_D_H1, T_max=10, eta_min=0.00001)
    CosineLR7 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_D_H2, T_max=10, eta_min=0.00001)

    data = faces_data(High_Data, Low_Data)
    sampler = torch.utils.data.SubsetRandomSampler(indices=list(range(100)))
    loader = DataLoader(dataset=data, batch_size=64, shuffle=True)
    num_test = 10
    data_name = "FFHQ"
    test_loader = get_loader(data_name, opt.batchsize)

    test_save = "./train/test_results"
    if not os.path.exists(test_save):
        os.makedirs(test_save)
    models_save = "./train/models"
    if not os.path.exists(models_save):
        os.makedirs(models_save)
    print('========================START TRAINING============================')
    for ep in tqdm(range(1, max_epoch+1)):
        T1 = time.time()
        G_DHL.train()
        G_RLS.train()
        G_DSL.train()
        D_L1.train()
        D_L2.train()
        D_H1.train()
        D_H2.train()

        for i, batch in enumerate(loader):
            print("\n Epoch & batch:", ep, i+1)
            optim_G_DHL.zero_grad()
            optim_G_RLS.zero_grad()
            optim_G_DSL.zero_grad()
            optim_D_L1.zero_grad()
            optim_D_L2.zero_grad()
            optim_D_H1.zero_grad()
            optim_D_H2.zero_grad()

            zs = batch["z"].cuda()
            lrs = batch["lr"].cuda()
            hrs = batch["hr"].cuda()
            downs = batch["hr_down"].cuda()

            lr_gen = G_DHL(hrs, zs)
            lr_gen_detach = lr_gen.detach()
            hr_gen = G_RLS(lr_gen_detach)
            hr_gen_detach = hr_gen.detach()
            hr_gen_r = G_RLS(lrs)
            hr_gen_r_detach = hr_gen_r.detach()
            lr_real_2 = G_DSL(hr_gen_r_detach, zs)
            lr_real_2_detach = lr_real_2.detach()
            ups = F.interpolate(lrs, size=[64, 64], mode="bicubic", align_corners=True)
            ups_detach = ups.detach()

            # update discriminators

            optim_D_L1.zero_grad()
            optim_D_L2.zero_grad()
            optim_D_H1.zero_grad()
            optim_D_H2.zero_grad()
            loss_D_L1 = nn.ReLU()(1.0 - D_L1(lrs)).mean() + nn.ReLU()(1 + D_L1(lr_gen_detach)).mean()
            loss_D_H1 = nn.ReLU()(1.0 - D_H1(hrs)).mean() + nn.ReLU()(1 + D_H1(hr_gen_detach)).mean()
            loss_D_L2 = nn.ReLU()(1.0 - D_L2(lrs)).mean() + nn.ReLU()(1 + D_L2(lr_real_2_detach)).mean()
            loss_D_H2 = nn.ReLU()(1.0 - D_H2(hrs)).mean() + nn.ReLU()(1 + D_H2(hr_gen_r_detach)).mean()
            loss_D_L1.backward()
            loss_D_L2.backward()
            loss_D_H1.backward()
            loss_D_H2.backward()
            optim_D_L1.step()
            optim_D_L2.step()
            optim_D_H1.step()
            optim_D_H2.step()

            # update generators
            # G_DHL

            optim_G_DHL.zero_grad()
            optim_G_RLS.zero_grad()
            optim_G_DSL.zero_grad()
            gan_loss_DHL = -D_L1(lr_gen).mean()
            L1_loss_DHL = L1Loss(lr_gen, downs)
            loss_G_DHL = alpha * L1_loss_DHL + beta * gan_loss_DHL
            loss_G_DHL.backward()
            optim_G_DHL.step()

            # G_RLS

            gan_loss_RLS_1 = -D_H1(hr_gen).mean()
            gan_loss_RLS_2 = -D_H2(hr_gen_r).mean()
            Cyc_loss_RLS_1 = L1Loss(hr_gen, hrs)
            L1_loss_RLS_2 = L1Loss(ups, hr_gen_r)
            loss_G_RLS_1 = alpha * Cyc_loss_RLS_1 + beta * gan_loss_RLS_1
            loss_G_RLS_2 = alpha * L1_loss_RLS_2 + beta * gan_loss_RLS_2
            loss_G_RLS = theta * loss_G_RLS_1 + gamma * loss_G_RLS_2
            loss_G_RLS.backward()
            optim_G_RLS.step()

            # G_DSL

            gan_loss_G_DSL = -D_L2(lr_real_2).mean()
            Cyc_loss_DSL = L1Loss(lr_real_2, lrs)
            loss_G_DSL = alpha * Cyc_loss_DSL + beta * gan_loss_G_DSL
            loss_G_DSL.backward()
            optim_G_DSL.step()

            lr1 = next(iter(optim_G_DHL.param_groups))['lr']
            lr2 = next(iter(optim_G_RLS.param_groups))['lr']
            lr3 = next(iter(optim_G_DSL.param_groups))['lr']
            lr4 = next(iter(optim_D_L1.param_groups))['lr']
            lr5 = next(iter(optim_D_L2.param_groups))['lr']
            lr6 = next(iter(optim_D_H1.param_groups))['lr']
            lr7 = next(iter(optim_D_H2.param_groups))['lr']

        writer.add_scalar('Train/loss_G_DHL', loss_G_DHL, ep)
        writer.add_scalar('Train/loss_G_RLS', loss_G_RLS, ep)
        writer.add_scalar('Train/loss_G_RLS_1', loss_G_RLS_1, ep)
        writer.add_scalar('Train/loss_G_RLS_2', loss_G_RLS_2, ep)
        writer.add_scalar('Train/loss_G_DSL', loss_G_DSL, ep)
        writer.add_scalar('Train/learnrate', lr1, ep)

        T2 = time.time()
        print(T2-T1)
        print("\n Testing and saving...")
        G_RLS.eval()
        for i, sample in enumerate(test_loader):
            if i >= num_test:
                break
            low_temp = sample["img16"].numpy()
            low = torch.from_numpy(np.ascontiguousarray(low_temp[:, ::-1, :, :])).cuda()
            with torch.no_grad():
                test_sr = G_RLS(low)
            test_lr = low.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            test_sr = test_sr.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            test_lr = (test_lr - test_lr.min()) / (test_lr.max() - test_lr.min())
            test_sr = (test_sr - test_sr.min()) / (test_sr.max() - test_sr.min())
            test_lr = (test_lr * 255).astype(np.uint8)
            test_sr = (test_sr * 255).astype(np.uint8)
            img_name = sample['imgpath'][0].split('/')[-1]
            cv2.imwrite("{}/{}_{:03d}_lr.png".format(test_save, img_name, ep), test_lr)
            cv2.imwrite("{}/{}_{:03d}_sr.png".format(test_save, img_name, ep), test_sr)
        if ep > 0:
            model_save = "{}/model_epoch_{:03d}.pth".format(models_save, ep)
            torch.save({"G_RLS": G_RLS.state_dict()}, model_save)
            print("saved: ", model_save)

        CosineLR1.step()
        CosineLR2.step()
        CosineLR3.step()
        CosineLR4.step()
        CosineLR5.step()
        CosineLR6.step()
        CosineLR7.step()

    writer.close()
    print("========================FINISHED============================")
