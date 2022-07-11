import argparse
import json
import logging
import random
from importlib import import_module
from os.path import join
import timeit
import numpy as np
import torch
from torch.distributions import Beta
from torch.utils.data import DataLoader
import open3d as o3d
import os

import sys
sys.path.append('../')
sys.path.append('./')

from utils.pcutil import plot_3d_point_cloud
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging

def addnoise(pcd):
    noise = (np.random.normal(0, 5, size=pcd.shape)/100.0).astype(np.float32)
    return pcd + noise

def downsample(pcd):
    for i in range(pcd.shape[2]):
        if i%2:
            pcd[:,:,i]=0
    return pcd

def downsample_fill(pcd):
    for i in range(pcd.shape[2]):
        if i%2:
            pcd[:,:,i]=pcd[:,:,i-1]
    return pcd

def main(eval_config):
    # Load hyperparameters as they were during training
    train_results_path = join(eval_config['results_root'], eval_config['arch'],
                              eval_config['experiment_name'])
    with open(join(train_results_path, 'config.json')) as f:
        train_config = json.load(f)

    random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])

    setup_logging(join(train_results_path, 'results'))
    log = logging.getLogger(__name__)

    weights_path = join(train_results_path, 'weights')
    if eval_config['max_epochs'] == 0:
        epoch = find_latest_epoch(weights_path)
    else:
        epoch = eval_config['max_epochs']
    log.debug(f'Starting from epoch: {epoch}')

    device = cuda_setup(eval_config['cuda'], eval_config['gpu'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    #
    # Dataset
    #
    dataset_name = train_config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=train_config['data_dir'],
                                  classes=train_config['classes'])
    elif dataset_name == 'modelnet':
        from datasets.datasetloader_pcd import DatasetLoaderPCD
        dataset = DatasetLoaderPCD(root=train_config['data_dir'])
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
                         
    classes_selected = ('all' if not train_config['classes']
                        else ','.join(train_config['classes']))
    log.debug(f'Selected {classes_selected} classes. Loaded {len(dataset)} '
              f'samples.')

    if 'distribution' in train_config:
        distribution = train_config['distribution']
    elif 'distribution' in eval_config:
        distribution = eval_config['distribution']
    else:
        log.warning('No distribution type specified. Assumed normal = N(0, 0.2)')
        distribution = 'normal'

    if train_config['reconstruction_loss'].lower() == 'chamfer':
        from losses.champfer_loss import ChamferLoss
        reconstruction_loss = ChamferLoss().to(device)
    elif train_config['reconstruction_loss'].lower() == 'earth_mover':
        from losses.earth_mover_distance import EMD
        reconstruction_loss = EMD().to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {train_config["reconstruction_loss"]}')

    #
    # Models
    #
    arch = import_module(f"models.{eval_config['arch']}")
    E = arch.Encoder(train_config).to(device)
    G = arch.Generator(train_config).to(device)

    #
    # Load saved state
    #
    E.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_E.pth')))
    G.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_G.pth')))

    E.eval()
    G.eval()

    # num_samples = len(dataset.point_clouds_names_test)
    data_loader = DataLoader(dataset, batch_size=train_config['batch_size'],
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    # We take 3 times as many samples as there are in test data in order to
    # perform JSD calculation in the same manner as in the reference publication
    # noise = torch.FloatTensor(3 * num_samples, train_config['z_size'], 1)
    # noise = noise.to(device)
    duration = 0
    loss_cd_all = 0
    nr=0
    for i, point_data in enumerate(data_loader):
        nr = nr+1
        X = point_data
        X_gt = X.clone()
        X_n=X.clone()
        if not eval_config['noise']=='none':
            if eval_config['noise']=='gauss':
                print("Applying gaussian noise.")
                X_n=addnoise(X_n)
            if eval_config['noise']=='dszero':
                print("Adding downsampling with zero noise to input.")
                X_n=downsample(X_n)
            if eval_config['noise']=='dscopy':
                print("Adding downsampling with copy noise to input.")
                X_n=downsample_fill(X_n)
        X_n = X_n.to(device)
        X_gt = X_gt.to(device)
        
        # np.save(join(train_results_path, 'results', f'{epoch:05}_X'), X.data.cpu().numpy())
        # np.save(join(train_results_path, 'results', f'{epoch:05}_X_noise'), X_n.data.cpu().numpy())

        # print(X_gt.shape)
        
        with torch.no_grad():
            start = timeit.default_timer()
            z_e = E(X_n)
            if isinstance(z_e, tuple):
                z_e = z_e[0]
            X_rec = G(z_e)
            stop = timeit.default_timer()
            if not ( i == 0 ):
                duration = duration + stop-start
        
        
        if X_gt.shape[-2:] == (3,2048):
            X_gt.transpose_(1, 2)
        if X_n.shape[-2:] == (3,2048):
            X_n.transpose_(1, 2)
        if X_rec.shape[-2:] == (3,2048):
            X_rec.transpose_(1, 2)
        X_rec = X_rec - X_rec.min()
        X_rec = X_rec/torch.abs(X_rec).max()
        loss_e = reconstruction_loss(X_gt,X_rec)
        loss_cd_all = loss_cd_all+loss_e
        if i == 0:
            gt = X_gt[0].detach().cpu().numpy().astype(np.float32)
            while len(gt.shape)>3:
                    gt=gt.squeeze(axis=0)
            gt=np.array(gt)
            # print(gt.shape)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(gt)
            path = os.path.join(train_results_path, 'results', f'{epoch:05}_X.pcd') 
            o3d.io.write_point_cloud(path,pcd)

            gt_noise = X_n[0].detach().cpu().numpy().astype(np.float32)
            while len(gt_noise.shape)>3:
                    gt_noise=gt_noise.squeeze(axis=0)
            gt_noise=np.array(gt_noise)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(gt_noise)
            path = os.path.join(train_results_path, 'results', f'{epoch:05}_X_noise.pcd') 
            o3d.io.write_point_cloud(path,pcd)
            
            gt_rec = X_rec[0].detach().cpu().numpy().astype(np.float32)
            while len(gt_rec.shape)>3:
                    gt_rec=gt_rec.squeeze(axis=0)
            gt_rec=np.array(gt_rec)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(gt_rec)
            path = os.path.join(train_results_path, 'results', f'{epoch:05}_X_rec.pcd') 
            o3d.io.write_point_cloud(path,pcd)
            # np.save(join(train_results_path, 'results', f'{epoch:05}_Xrec'), X_rec.data.cpu().numpy())
    print('Time per batch is: {} sec.'.format(duration/nr))
    print('Time per image is: {} sec.'.format(duration/nr/train_config['batch_size']))
    print('Chamfer distance loss per image: {}'.format(loss_cd_all/nr/train_config['batch_size']))

if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='File path for evaluation config')
    args = parser.parse_args()

    evaluation_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            evaluation_config = json.load(f)
    assert evaluation_config is not None

    main(evaluation_config)
