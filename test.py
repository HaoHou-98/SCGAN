import os
os.sys.path.append(os.getcwd())
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
from data_test import get_loader
from torch.autograd import Variable
import torchvision.utils as vutils
from model import G_RLS
import cv2
import time

def to_var(data):
    real_cpu = data
    batchsize = real_cpu.size(0)
    input = Variable(real_cpu.cuda())
    return input, batchsize

def main():
    torch.manual_seed(1)
    np.random.seed(0)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    opt = edict()
    opt.nGPU = 1
    opt.batchsize = 1
    opt.cuda = True
    cudnn.benchmark = True
    print('========================LOAD DATA============================')
    data_name = 'FFHQ'
    test_loader = get_loader(data_name, opt.batchsize)
    net_G_RLS = G_RLS()
    net_G_RLS = net_G_RLS.cuda()
    a = torch.load('./pretrained_model/pretrained_model.pth')["G_h2l"]
    net_G_RLS.load_state_dict(a)
    net_G_RLS = net_G_RLS.eval()
    test_save = './test_results'
    if not os.path.exists(test_save):
        os.makedirs(test_save)
    for i, sample in enumerate(test_loader):
        print(i)
        low_temp = sample["img16"].numpy()
        low = torch.from_numpy(np.ascontiguousarray(low_temp[:, ::-1, :, :])).cuda()
        with torch.no_grad():
            test_sr = net_G_RLS(low)
        test_low = low.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        test_sr = test_sr.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        test_low = (test_low - test_low.min()) / (test_low.max() - test_low.min())
        test_sr = (test_sr - test_sr.min()) / (test_sr.max() - test_sr.min())
        test_low = (test_low * 255).astype(np.uint8)
        test_sr = (test_sr * 255).astype(np.uint8)
        img_name = sample['imgpath'][0].split('/')[-1]
        cv2.imwrite("{}/{}_lr.png".format(test_save, img_name), test_low)
        cv2.imwrite("{}/{}_sr.png".format(test_save, img_name), test_sr)

if __name__ == '__main__':
    main()
