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
import cv2
import numpy as np
sys.path.append('../')
sys.path.append('./')
from utils.pcutil import plot_3d_point_cloud
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging
import open3d as o3d
import trimesh
from losses.losses2d import *

cudnn.benchmark = True


def gim2pcd(gim):
    if len(gim.shape)>3:
        gimpcd=np.empty([gim.shape[0],3,2048])
        for j in range(gim.shape[0]):
            gimnp=np.array(gim[j])
            gim2flat = np.array([gimnp[0].flatten(),gimnp[1].flatten(),gimnp[2].flatten()])
            gim2flat = np.moveaxis(gim2flat,0,-1)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(gim2flat)
            pcd.estimate_normals()
            # estimate radius for rolling ball
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 1.5 * avg_dist
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd,
                    o3d.utility.DoubleVector([radius, radius * 2]))
            # create the triangular mesh with the vertices and faces from open3d
            tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                    vertex_normals=np.asarray(mesh.vertex_normals))

            trimesh.convex.is_convex(tri_mesh)
            points=tri_mesh.sample(2048)
            pointsnp=np.array(points)
            pointsnp = np.moveaxis(pointsnp,0,-1)
            pointsnp = pointsnp - pointsnp.min()
            pointsnp_normalized = pointsnp/np.absolute(pointsnp).max()-0.5
            gimpcd[j]=pointsnp_normalized
        return torch.from_numpy(gimpcd)
    else:
        return

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
        loss_gim = 0
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'])
    elif dataset_name == 'faust':
        from datasets.dfaust import DFaustDataset
        dataset = DFaustDataset(root_dir=config['data_dir'],
                                classes=config['classes'])
    elif dataset_name == 'modelnet':
        from datasets.datasetloader import DatasetLoader
        dataset = DatasetLoader(root=config['data_dir'],
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
    G = arch.Generator(config).to(device)
    E = arch.Encoder(config).to(device)
    D = arch.Discriminator(config).to(device)

    G.apply(weights_init)
    E.apply(weights_init)
    D.apply(weights_init)

    if config['reconstruction_loss'].lower() == 'chamfer':
        from losses.champfer_loss import ChamferLoss
        reconstruction_loss = ChamferLoss().to(device)
    elif config['reconstruction_loss'].lower() == 'earth_mover':
        from losses.earth_mover_distance import EMD
        reconstruction_loss = EMD().to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')
    rmse = RMSE()
    depth_criterion = RMSE_log()
    grad_criterion = GradLoss()
    normal_criterion = NormalLoss()
    eval_metric = RMSE_log()
    l1_criterion = L1()
    grad_factor = 10.
    normal_factor = 1.
    gim_factor = 10.
    #
    # Float Tensors
    #
    # fixed_noise = torch.FloatTensor(config['batch_size'], 3,config['latent_image_height'], config['latent_image_width'])
    # fixed_noise.normal_(mean=config['normal_mu'], std=config['normal_std'])
    # print(fixed_noise.min())
    # print(fixed_noise.max())
    noise = torch.FloatTensor(config['batch_size'], 3,config['latent_image_height'], config['latent_image_width'])
    # noise = torch.FloatTensor(config['batch_size'], config['z_size'])
    # fixed_noise = fixed_noise.to(device)
    noise = noise.to(device)

    #
    # Optimizers
    #
    EG_optim = getattr(optim, config['optimizer']['EG']['type'])
    EG_optim = EG_optim(chain(E.parameters(), G.parameters()),
                        **config['optimizer']['EG']['hyperparams'])

    D_optim = getattr(optim, config['optimizer']['D']['type'])
    D_optim = D_optim(D.parameters(),
                      **config['optimizer']['D']['hyperparams'])

    if starting_epoch > 1:
        G.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_G.pth')))
        E.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_E.pth')))
        D.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_D.pth')))

        D_optim.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_Do.pth')))

        EG_optim.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch-1:05}_EGo.pth')))

    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()

        G.train()
        E.train()
        D.train()

        total_loss_d = 0.0
        total_loss_eg = 0.0
        for i, point_data in enumerate(points_dataloader, 1):
            log.debug('-' * 20)
            X, gimgt = point_data
            X = X.to(device)
            gimgt = gimgt.to(device)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)

            codes, _, _ = E(X)
            if dataset_name == 'modelnet':
                depth_loss = depth_criterion(codes, gimgt)
                # grad_real, grad_fake = imgrad_yx(gimgt), imgrad_yx(codes)
                # grad_loss = grad_criterion(grad_fake, grad_real)     * grad_factor * (epoch>3)
                # normal_loss = normal_criterion(grad_fake, grad_real) * normal_factor * (epoch>7)
                l1_loss = l1_criterion(codes, gimgt)
                loss_gim = depth_loss + l1_loss#+ grad_loss + normal_loss
                loss_gim*=gim_factor

            useD=False
            if useD:
                noise.normal_(mean=config['normal_mu'], std=config['normal_std'])
                synth_logit = D(codes)
                real_logit = D(noise)
                loss_d = torch.mean(synth_logit) - torch.mean(real_logit)

                alpha = torch.rand(config['batch_size'], 3,config['latent_image_height'], config['latent_image_width']).to(device)
                # alpha = torch.rand(config['batch_size'], config['z_size']).to(device)
                differences = codes - noise
                interpolates = noise + alpha * differences
                disc_interpolates = D(interpolates)

                gradients = grad(
                    outputs=disc_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones_like(disc_interpolates).to(device),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
                slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
                gradient_penalty = ((slopes - 1) ** 2).mean()
                loss_gp = torch.sqrt(config['gp_lambda'] * gradient_penalty)
                ###
                loss_d += loss_gp
                # loss_d *= 0.01
                D_optim.zero_grad()
                D.zero_grad()

                loss_d.backward(retain_graph=True)
                total_loss_d += loss_d.item()
                D_optim.step()

            # EG part of training
            X_rec = G(codes)

            loss_e = torch.sum(
                config['reconstruction_coef'] *
                reconstruction_loss(X.permute(0, 2, 1) + 0.5,
                                    X_rec.permute(0, 2, 1) + 0.5))

            if useD:
                synth_logit = D(codes)
                loss_g = -torch.mean(synth_logit)
                loss_eg = loss_e + loss_gim# + loss_g
            else:
                loss_eg = loss_e + loss_gim

            EG_optim.zero_grad()
            E.zero_grad()
            G.zero_grad()

            loss_eg.backward()
            total_loss_eg += loss_eg.item()
            EG_optim.step()

            print(f'[{epoch}: ({i})] '
                    #   f'Loss_D: {loss_d:.4f} '
                    #   f'(Loss_GP: {loss_gp: .4f}) '
                      f'Loss_EG: {loss_eg:.4f} '
                      f'(Loss_E: {loss_e: .4f}) '
                    #   f'(Loss_G: {loss_g: .4f}) '
                      f'(Loss_GIM: {loss_gim: .4f}) '
                      f'Time: {datetime.now() - start_epoch_time}')

        print(
            f'[{epoch}/{config["max_epochs"]}] '
            f'Loss_D: {total_loss_d / i:.4f} '
            f'Loss_EG: {total_loss_eg / i:.4f} '
            f'Time: {datetime.now() - start_epoch_time}'
        )

        #
        # Save intermediate results
        #
        G.eval()
        E.eval()
        D.eval()
        with torch.no_grad():
            # fake = G(fixed_noise).data.cpu().numpy()
            codes, _, _ = E(X)
            X_rec = G(codes).data.cpu().numpy()
            latentrgb2np = codes.cpu().detach().numpy()   

        for k in range(5):
            # print(X.shape)
            # print(X[k].shape)
            fig = plot_3d_point_cloud(X[k][0], X[k][1], X[k][2],
                                      in_u_sphere=True, show=False,
                                      title=str(epoch))
            fig.savefig(
                join(results_dir, 'samples', f'{epoch:05}_{k}_real.png'))
            plt.close(fig)
        
        if epoch%200==0:
            for k in range(5):
                print(f'gim2pcd_{k}')
                fakepcd=gim2pcd(latentrgb2np)
                fig = plot_3d_point_cloud(fakepcd[0], fakepcd[1], fakepcd[2],
                                        in_u_sphere=True, show=False,
                                        title=str(epoch))
                fig.savefig(
                    join(results_dir, 'samples', f'{epoch:05}_{k}_fake.png'))
                plt.close(fig)

        for k in range(5):
            latentrgb2npk=latentrgb2np[k]
            latentrgb2npk=latentrgb2npk*255
            latentrgb2npk = np.moveaxis(latentrgb2npk,0,-1)
            path = join(results_dir, 'samples', f'{epoch:05}_{k}_latentrgb.png')
            cv2.imwrite(path,latentrgb2npk)

        # for k in range(5):
        #     # print(gimgt.shape)
        #     latentrgb2npk=gimgt[k].cpu().detach().numpy()
        #     latentrgb2npk=latentrgb2npk*255
        #     latentrgb2npk = np.moveaxis(latentrgb2npk,0,-1)
        #     path = join(results_dir, 'samples', f'{epoch:05}_{k}_gimgt.png')
        #     cv2.imwrite(path,latentrgb2npk)

        # for k in range(5):
        #     fig = plot_3d_point_cloud(fake[k][0], fake[k][1], fake[k][2],
        #                               in_u_sphere=True, show=False,
        #                               title=str(epoch))
        #     fig.savefig(
        #         join(results_dir, 'samples', f'{epoch:05}_{k}_fixed.png'))
        #     plt.close(fig)

        for k in range(5):
            fig = plot_3d_point_cloud(X_rec[k][0],
                                      X_rec[k][1],
                                      X_rec[k][2],
                                      in_u_sphere=True, show=False,
                                      title=str(epoch))
            fig.savefig(join(results_dir, 'samples',
                             f'{epoch:05}_{k}_reconstructed.png'))
            plt.close(fig)

        if epoch % config['save_frequency'] == 0:
            torch.save(G.state_dict(), join(weights_path, f'{epoch:05}_G.pth'))
            torch.save(D.state_dict(), join(weights_path, f'{epoch:05}_D.pth'))
            torch.save(E.state_dict(), join(weights_path, f'{epoch:05}_E.pth'))

            torch.save(EG_optim.state_dict(),
                       join(weights_path, f'{epoch:05}_EGo.pth'))

            torch.save(D_optim.state_dict(),
                       join(weights_path, f'{epoch:05}_Do.pth'))


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
