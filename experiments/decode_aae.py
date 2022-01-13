import argparse
import json
import logging
import random
from datetime import datetime
from importlib import import_module
from itertools import chain
from os.path import join, exists

import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import sys
import cv2
import os
import numpy as np
sys.path.append('../')
sys.path.append('./')
from utils.pcutil import plot_3d_point_cloud
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging

cudnn.benchmark = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def main(config):
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    results_dir = prepare_results_dir(config)
    starting_epoch = find_latest_epoch(results_dir) + 1

    if not exists(join(results_dir, 'config.json')):
        with open(join(results_dir, 'config.json'), mode='w') as f:
            json.dump(config, f)

    setup_logging(results_dir)
    log = logging.getLogger(__name__)

    device = cuda_setup(config['cuda'], config['gpu'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    weights_path = join(results_dir, 'weights')

    #
    # Dataset
    #
    # dataset_name = config['dataset'].lower()
    # if dataset_name == 'shapenet':
    #     from datasets.shapenet import ShapeNetDataset
    #     dataset = ShapeNetDataset(root_dir=config['data_dir'],
    #                               classes=config['classes'])
    # elif dataset_name == 'faust':
    #     from datasets.dfaust import DFaustDataset
    #     dataset = DFaustDataset(root_dir=config['data_dir'],
    #                             classes=config['classes'])
    # else:
    #     raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
    #                      f'`faust`. Got: `{dataset_name}`')
    # log.debug("Selected {} classes. Loaded {} samples.".format(
    #     'all' if not config['classes'] else ','.join(config['classes']),
    #     len(dataset)))

    # points_dataloader = DataLoader(dataset, batch_size=config['batch_size'],
    #                                shuffle=config['shuffle'],
    #                                num_workers=config['num_workers'],
    #                                drop_last=True, pin_memory=True)

    #
    # Models
    #
    arch = import_module(f"models.{config['arch']}")
    D = arch.Discriminator(config).to(device)

    D.apply(weights_init)

    #
    # Float Tensors
    #
    fixed_noise = torch.FloatTensor(config['batch_size'], config['z_size'], 1)
    fixed_noise.normal_(mean=config['normal_mu'], std=config['normal_std'])
    noise = torch.FloatTensor(config['batch_size'], config['z_size'])

    fixed_noise = fixed_noise.to(device)
    noise = noise.to(device)

    D_optim = getattr(optim, config['optimizer']['D']['type'])
    D_optim = D_optim(D.parameters(),
                      **config['optimizer']['D']['hyperparams'])

    if starting_epoch > 1:
        D.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_D.pth')))

        D_optim.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_Do.pth')))

    
    D.eval()
    with torch.no_grad():

        dlist=os.listdir(config['dec_path'])
        dlist.sort()
        # time_sum = 0
        # counter = 0
        for filename in dlist:
            if filename.endswith(".png"):
                path=config['dec_path']+"/"+filename
                print("Predicting for:"+filename)
                image = cv2.imread(path,cv2.IMREAD_UNCHANGED ).astype(np.float32)
                image = torch.from_numpy(np.moveaxis(image,-1,0))
                image_batch =torch.empty((50,3,64,64), dtype=torch.float32)
                for i in range(50):
                    image_batch[i]=image
                print(image_batch.shape)
                # img = torch.from_numpy(depth2).float().unsqueeze(0)
                # m_depth=torch.max(img)
                # img=img/m_depth
                # start = timeit.default_timer()
                z_fake = D(image_batch.to(device))
                # stop = timeit.default_timer()
                # time_sum=time_sum+stop-start
                # counter=counter+1
                # zfv=z_fake*2-1
                # z_fake_norm=zfv.pow(2).sum(dim=1).pow(0.5).unsqueeze(1)
                # zfv=zfv/z_fake_norm
                # z_fake=(zfv+1)/2
                save_path=path[:-4]
                save_image(z_fake[0], save_path +"_dec"+'.png')
            else:
                continue

if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    main(config)
