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
import sys
import numpy as np
import cv2
sys.path.append('../')
sys.path.append('./')
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
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'])
    elif dataset_name == 'faust':
        from datasets.dfaust import DFaustDataset
        dataset = DFaustDataset(root_dir=config['data_dir'],
                                classes=config['classes'])
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
    log.debug("Selected {} classes. Loaded {} samples.".format(
        'all' if not config['classes'] else ','.join(config['classes']),
        len(dataset)))

    points_dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                   shuffle=config['shuffle'],
                                   num_workers=config['num_workers'],
                                   drop_last=True, pin_memory=True)

    #
    # Models
    #
    
    arch = import_module(f"models.{config['arch']}")
    E = arch.Encoder(config).to(device)
    E.apply(weights_init)
    G = arch.Generator(config).to(device)
    EG_optim = getattr(optim, config['optimizer']['EG']['type'])
    EG_optim = EG_optim(chain(E.parameters(), G.parameters()),
                        **config['optimizer']['EG']['hyperparams'])
    # EG_optim = EG_optim(E.parameters(),
    #                     **config['optimizer']['EG']['hyperparams'])
    E.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_E.pth')))
    EG_optim.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_EGo.pth')))
   
    for i, point_data in enumerate(points_dataloader, 1):
        start_epoch_time = datetime.now()
        X, _ = point_data
        X = X.to(device)
        if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)
        E.eval()
        with torch.no_grad():
            _, _, _, latentrgb = E(X)
        bs = config['batch_size']
        for j in range(bs):
            nr = (i-1)*bs+j
            # print(codes[j].squeeze(dim=0).shape)
            latentrgb2np = latentrgb.squeeze(dim=0).cpu().detach().numpy()
            latentrgb2npk=latentrgb2np[j]
            latentrgb2npk=latentrgb2npk*255/latentrgb2npk.max()
            # print(latentrgb2npk.shape)
            latentrgb2npk = np.moveaxis(latentrgb2npk,0,-1)
            # print(latentrgb2npk.shape)
            # im = Image.fromarray(latentrgb2np)
            path = join(results_dir, 'enc_dataset', f'{nr:05}.png')
            # print(path)
            cv2.imwrite(path,latentrgb2npk)


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
