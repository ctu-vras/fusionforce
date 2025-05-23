#!/usr/bin/env python

import sys
sys.path.append('../src')
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from fusionforce.models.terrain_encoder.utils import denormalize_img, ego_to_cam, get_only_in_img_mask
from fusionforce.models.terrain_encoder.lss import LiftSplatShoot
from fusionforce.models.terrain_encoder.voxelnet import VoxelNet
from fusionforce.models.terrain_encoder.bevfusion import BEVFusion
from fusionforce.models.traj_predictor.dphysics import DPhysics
from fusionforce.models.traj_predictor.dphys_config import DPhysConfig
from fusionforce.datasets.rough import ROUGH
from fusionforce.utils import read_yaml, write_to_yaml, str2bool, compile_data, position
from fusionforce.losses import hm_loss, physics_loss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='Train MonoForce model')
    parser.add_argument('--model', type=str, default='lss', help='Model to train: lss, voxelnet, bevfusion')
    parser.add_argument('--bsz', type=int, default=4, help='Batch size')
    parser.add_argument('--nepochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--robot', type=str, default='marv', help='Robot name')
    parser.add_argument('--lss_cfg_path', type=str, default='../config/lss_cfg.yaml', help='Path to LSS config')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--debug', type=str2bool, default=True, help='Debug mode: use small datasets')
    parser.add_argument('--vis', type=str2bool, default=False, help='Visualize training samples')
    parser.add_argument('--geom_weight', type=float, default=1.0, help='Weight for geometry loss')
    parser.add_argument('--terrain_weight', type=float, default=2.0, help='Weight for terrain heightmap loss')
    parser.add_argument('--phys_weight', type=float, default=1.0, help='Weight for physics loss')
    parser.add_argument('--traj_sim_time', type=float, default=5.0, help='Trajectory simulation time')
    parser.add_argument('--dphys_grid_res', type=float, default=0.4, help='DPhys grid resolution')

    return parser.parse_args()


class TrainerCore:
    """
    Trainer for terrain encoder model
    """

    def __init__(self,
                 dphys_cfg,
                 lss_cfg,
                 model,
                 nepochs=1000,
                 debug=False,
                 geom_weight=1.0,
                 terrain_weight=1.0,
                 phys_weight=1.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'rough'
        self.model = model
        self.dphys_cfg = dphys_cfg
        self.lss_cfg = lss_cfg
        self.debug = debug

        self.nepochs = nepochs
        self.min_loss = np.inf
        self.min_train_loss = np.inf
        self.train_counter = 0
        self.val_counter = 0

        self.geom_weight = geom_weight
        self.terrain_weight = terrain_weight
        self.phys_weight = phys_weight

        # models and optimizer
        self.terrain_encoder = None
        self.dphysics = DPhysics(dphys_cfg, device=self.device)

        # coarser grid resolution for dphysics: average pooling of terrain encoder grid
        self.dphys_grid_res = self.dphys_cfg.grid_res
        self.terrain_encoder_grid_res = self.lss_cfg['grid_conf']['xbound'][2]
        kernel_size = int(self.dphys_grid_res / self.terrain_encoder_grid_res)
        self.terrain_preproc = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)

        # for terrain regularization: penalize Laplacian of the terrain (2nd derivative of elevation)
        self.laplace_kernel = torch.tensor([[1, 1, 1],
                                            [1, -8, 1],
                                            [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        # optimizer
        self.optimizer = None
        
        # dataloaders
        self.train_loader = None
        self.val_loader = None

        self.log_dir = os.path.join('../config/tb_runs/',
                                    f'{self.dataset}/{self.model}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def create_dataloaders(self, bsz=1, debug=False, vis=False, Data=ROUGH):
        # create dataset for LSS model training
        train_ds, val_ds = compile_data(small_data=debug, vis=vis, Data=Data,
                                        dphys_cfg=self.dphys_cfg, lss_cfg=self.lss_cfg)

        # create dataloaders: making sure all elemts in a batch are tensors
        def collate_fn(batch):
            def to_tensor(item):
                if isinstance(item, np.ndarray):
                    return torch.tensor(item)
                elif isinstance(item, (list, tuple)):
                    return [to_tensor(i) for i in item]
                elif isinstance(item, dict):
                    return {k: to_tensor(v) for k, v in item.items()}
                return item  # Return as is if it's already a tensor or unsupported type

            batch = [to_tensor(item) for item in batch]
            return torch.utils.data.default_collate(batch)

        train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=bsz, shuffle=False, collate_fn=collate_fn)

        return train_loader, val_loader

    def compute_losses(self, batch):
        losses = {}
        return losses

    def laplacian_loss(self, pred):
        lap = torch.nn.functional.conv2d(pred, self.laplace_kernel, padding=1)
        return torch.mean(torch.abs(lap))

    def epoch(self, train=True):
        loader = self.train_loader if train else self.val_loader
        counter = self.train_counter if train else self.val_counter

        if train:
            self.terrain_encoder.train()
        else:
            self.terrain_encoder.eval()

        max_grad_norm = 1.0
        epoch_losses = {}
        for batch in tqdm(loader, total=len(loader)):
            if train:
                self.optimizer.zero_grad()

            batch = [torch.as_tensor(b, dtype=torch.float32, device=self.device) for b in batch]
            losses = self.compute_losses(batch)
            loss = (self.geom_weight * losses['geom'] +
                    self.terrain_weight * losses['terrain'] +
                    self.phys_weight * losses['phys'])

            if torch.isnan(loss):
                torch.save(self.terrain_encoder.state_dict(), os.path.join(self.log_dir, 'train.pth'))
                raise ValueError('Loss is NaN')

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.terrain_encoder.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()

            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()
            epoch_losses['total'] += (losses['geom'] + losses['terrain'] + losses['phys']).item()

            counter += 1
            for k, v in losses.items():
                self.writer.add_scalar(f"{'train' if train else 'val'}/iter_loss_{k}", v.item(), counter)

        if len(loader) > 0:
            for k, v in epoch_losses.items():
                epoch_losses[k] /= len(loader)

        return epoch_losses, counter

    def train(self):
        # save configs to log dir
        write_to_yaml(self.dphys_cfg.__dict__, os.path.join(self.log_dir, 'dphys_cfg.yaml'))
        write_to_yaml(self.lss_cfg, os.path.join(self.log_dir, 'lss_cfg.yaml'))

        for e in range(self.nepochs):
            # training epoch
            train_losses, self.train_counter = self.epoch(train=True)
            for k, v in train_losses.items():
                print('Epoch:', e, f'Train loss {k}:', v)
                self.writer.add_scalar(f'train/epoch_loss_{k}', v, e)

            if train_losses['total'] < self.min_train_loss or self.debug:
                with torch.no_grad():
                    self.min_train_loss = train_losses['total']
                    print('Saving train model...')
                    self.terrain_encoder.eval()
                    torch.save(self.terrain_encoder.state_dict(), os.path.join(self.log_dir, 'train.pth'))

                    # visualize training predictions
                    fig = self.vis_pred(self.train_loader)
                    self.writer.add_figure('train/prediction', fig, e)

            # validation epoch
            with torch.inference_mode():
                with torch.no_grad():
                    val_losses, self.val_counter = self.epoch(train=False)
                    for k, v in val_losses.items():
                        print('Epoch:', e, f'Val loss {k}:', v)
                        self.writer.add_scalar(f'val/epoch_loss_{k}', v, e)

                    if val_losses['total'] < self.min_loss or self.debug:
                        self.min_loss = val_losses['total']
                        print('Saving model...')
                        self.terrain_encoder.eval()
                        torch.save(self.terrain_encoder.state_dict(), os.path.join(self.log_dir, 'val.pth'))

                        # visualize validation predictions
                        fig = self.vis_pred(self.val_loader)
                        self.writer.add_figure('val/prediction', fig, e)

    def pred(self, sample):
        raise NotImplementedError

    def predicts_states(self, terrain, pose0, controls):
        # preprocess terrain for physics
        terrain_ = {}
        for k, v in terrain.items():
            terrain_[k] = self.terrain_preproc(v)
        # initial state
        x0 = pose0[:, :3, 3].to(self.device)
        xd0 = torch.zeros_like(x0)
        R0 = pose0[:, :3, :3].to(self.device)
        omega0 = torch.zeros_like(xd0)
        state0 = (x0, xd0, R0, omega0)
        # predict states
        states_pred, _ = self.dphysics(z_grid=terrain_['terrain'].squeeze(1), state=state0,
                                       controls=controls.to(self.device),
                                       friction=terrain_['friction'].squeeze(1))
        return states_pred

    @torch.no_grad()
    def vis_pred(self, loader):
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        # visualize training predictions
        sample_i = np.random.choice(len(loader.dataset))
        sample = loader.dataset[sample_i]

        if self.model == 'lss':
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_geom, hm_terrain,
             controls_ts, controls,
             pose0,
             traj_ts, xs, xds, Rs, omegas) = sample
        elif self.model == 'bevfusion':
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_geom, hm_terrain,
             controls_ts, controls,
             pose0,
             traj_ts, xs, xds, Rs, omegas,
             points) = sample
        elif self.model == 'voxelnet':
            (points, hm_geom, hm_terrain,
             controls_ts, controls,
             pose0,
             traj_ts, xs, xds, Rs, omegas) = sample
        else:
            raise ValueError('Model not supported')

        # predict height maps and states
        with torch.no_grad():
            batch = [torch.as_tensor(b, dtype=torch.float32, device=self.device).unsqueeze(0) for b in sample]
            terrain, states_pred = self.pred(batch)

        geom_pred = terrain['geom'][0, 0].cpu()
        diff_pred = terrain['diff'][0, 0].cpu()
        terrain_pred = terrain['terrain'][0, 0].cpu()
        friction_pred = terrain['friction'][0, 0].cpu()
        Xs_pred = states_pred[0][0].cpu()
        Xs_pred_grid = (Xs_pred[:, :2] + self.dphys_cfg.d_max) / self.terrain_encoder_grid_res
        Xs_grid = (xs[:, :2] + self.dphys_cfg.d_max) / self.terrain_encoder_grid_res

        # get height map points
        z_grid = terrain_pred
        x_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.terrain_encoder_grid_res)
        y_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.terrain_encoder_grid_res)
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
        hm_points = torch.stack([x_grid, y_grid, z_grid], dim=-1)
        hm_points = hm_points.view(-1, 3).T

        if self.model in ['lss', 'bevfusion']:
            # plot images with projected height map points
            img_H, img_W = self.lss_cfg['data_aug_conf']['H'], self.lss_cfg['data_aug_conf']['W']
            for imgi in range(len(imgs))[:4]:
                ax = axes[0, imgi]
                img = imgs[imgi]
                img = denormalize_img(img[:3])

                cam_pts = ego_to_cam(hm_points, rots[imgi], trans[imgi], intrins[imgi])
                mask_img = get_only_in_img_mask(cam_pts, img_H, img_W)
                plot_pts = post_rots[imgi].matmul(cam_pts) + post_trans[imgi].unsqueeze(1)

                cam_pts_Xs = ego_to_cam(xs[:, :3].T, rots[imgi], trans[imgi], intrins[imgi])
                mask_img_Xs = get_only_in_img_mask(cam_pts_Xs, img_H, img_W)
                plot_pts_Xs = post_rots[imgi].matmul(cam_pts_Xs) + post_trans[imgi].unsqueeze(1)

                cam_pts_Xs_pred = ego_to_cam(Xs_pred[:, :3].T, rots[imgi], trans[imgi], intrins[imgi])
                mask_img_Xs_pred = get_only_in_img_mask(cam_pts_Xs_pred, img_H, img_W)
                plot_pts_Xs_pred = post_rots[imgi].matmul(cam_pts_Xs_pred) + post_trans[imgi].unsqueeze(1)

                ax.imshow(img)
                ax.scatter(plot_pts[0, mask_img], plot_pts[1, mask_img], s=1, c=hm_points[2, mask_img],
                           cmap='jet', vmin=-1.0, vmax=1.0)
                ax.scatter(plot_pts_Xs[0, mask_img_Xs], plot_pts_Xs[1, mask_img_Xs], c='k', s=1)
                ax.scatter(plot_pts_Xs_pred[0, mask_img_Xs_pred], plot_pts_Xs_pred[1, mask_img_Xs_pred], c='r', s=1)
                ax.axis('off')

        axes[1, 0].set_title('Prediction: Terrain')
        axes[1, 0].imshow(terrain_pred.T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)
        axes[1, 0].scatter(Xs_pred_grid[:, 0], Xs_pred_grid[:, 1], c='r', s=1)
        axes[1, 0].scatter(Xs_grid[:, 0], Xs_grid[:, 1], c='k', s=1)

        axes[1, 1].set_title('Label: Terrain')
        axes[1, 1].imshow(hm_terrain[0].T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)
        axes[1, 1].scatter(Xs_pred_grid[:, 0], Xs_pred_grid[:, 1], c='r', s=1)
        axes[1, 1].scatter(Xs_grid[:, 0], Xs_grid[:, 1], c='k', s=1)

        axes[1, 2].set_title('Friction')
        axes[1, 2].imshow(friction_pred.T, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)
        axes[1, 2].scatter(Xs_pred_grid[:, 0], Xs_pred_grid[:, 1], c='r', s=1)
        axes[1, 2].scatter(Xs_grid[:, 0], Xs_grid[:, 1], c='k', s=1)

        axes[1, 3].set_title('Trajectories XY')
        axes[1, 3].plot(xs[:, 0], xs[:, 1], c='k', label='GT')
        axes[1, 3].plot(Xs_pred[:, 0], Xs_pred[:, 1], c='r', label='Pred')
        axes[1, 3].set_xlabel('X [m]')
        axes[1, 3].set_ylabel('Y [m]')
        axes[1, 3].set_xlim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
        axes[1, 3].set_ylim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
        axes[1, 3].grid()
        axes[1, 3].legend()

        axes[2, 0].set_title('Prediction: Geom')
        axes[2, 0].imshow(geom_pred.T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)
        axes[2, 0].scatter(Xs_pred_grid[:, 0], Xs_pred_grid[:, 1], c='r', s=5)
        axes[2, 0].scatter(Xs_grid[:, 0], Xs_grid[:, 1], c='k', s=1)

        axes[2, 1].set_title('Label: Geom')
        axes[2, 1].imshow(hm_geom[0].T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)
        axes[2, 1].scatter(Xs_pred_grid[:, 0], Xs_pred_grid[:, 1], c='r', s=5)
        axes[2, 1].scatter(Xs_grid[:, 0], Xs_grid[:, 1], c='k' , s=1)

        axes[2, 2].set_title('Height diff')
        axes[2, 2].imshow(diff_pred.T, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)
        axes[2, 2].scatter(Xs_pred_grid[:, 0], Xs_pred_grid[:, 1], c='r', s=5)
        axes[2, 2].scatter(Xs_grid[:, 0], Xs_grid[:, 1], c='k' , s=1)

        axes[2, 3].set_title('Trajectories Z')
        axes[2, 3].plot(traj_ts, xs[:, 2], 'k', label='GT')
        axes[2, 3].plot(controls_ts, Xs_pred[:, 2], c='r', label='Pred')
        axes[2, 3].set_xlabel('Time [s]')
        axes[2, 3].set_ylabel('Z [m]')
        axes[2, 3].set_ylim(-self.dphys_cfg.h_max, self.dphys_cfg.h_max)
        axes[2, 3].grid()
        axes[2, 3].legend()

        return fig


class TrainerLSS(TrainerCore):
    def __init__(self, dphys_cfg, lss_cfg, model='lss', bsz=1, lr=1e-3, nepochs=1000,
                 pretrained_model_path=None, debug=False, vis=False, geom_weight=1.0, terrain_weight=1.0, phys_weight=1.0):
        super().__init__(dphys_cfg, lss_cfg, model, nepochs, debug, geom_weight, terrain_weight, phys_weight)
        # create dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders(bsz=bsz, debug=debug, vis=vis, Data=ROUGH)

        # load models: terrain encoder
        self.terrain_encoder = LiftSplatShoot(self.lss_cfg['grid_conf'],
                                              self.lss_cfg['data_aug_conf']).from_pretrained(pretrained_model_path)
        self.terrain_encoder.to(self.device)

        # define optimizer
        self.optimizer = torch.optim.Adam(self.terrain_encoder.parameters(),
                                          lr=lr, betas=(0.8, 0.999), weight_decay=1e-7)

    def compute_losses(self, batch):
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain,
         control_ts, controls,
         pose0,
         traj_ts, xs, xds, Rs, omegas) = batch
        # terrain encoder forward pass
        inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        terrain = self.terrain_encoder(*inputs)

        # geometry loss: difference between predicted and ground truth height maps
        if self.geom_weight > 0:
            loss_geom = hm_loss(terrain['geom'], hm_geom[:, 0:1], hm_geom[:, 1:2])
            loss_geom += 0.02 * self.laplacian_loss(terrain['geom'])
        else:
            loss_geom = torch.tensor(0.0, device=self.device)

        # rigid / terrain height map loss
        if self.terrain_weight > 0:
            loss_terrain = hm_loss(terrain['terrain'], hm_terrain[:, 0:1], hm_terrain[:, 1:2])
            loss_terrain += 0.02 * self.laplacian_loss(terrain['terrain'])
        else:
            loss_terrain = torch.tensor(0.0, device=self.device)

        # physics loss: difference between predicted and ground truth states
        if self.phys_weight > 0:
            # predict trajectory
            states_gt = [xs, xds, Rs, omegas]
            states_pred = self.predicts_states(terrain, pose0, controls)
            # compute physics loss
            loss_phys = physics_loss(states_pred=states_pred, states_gt=states_gt,
                                     pred_ts=control_ts, gt_ts=traj_ts)
        else:
            loss_phys = torch.tensor(0.0, device=self.device)

        losses = {
            'geom': loss_geom,
            'terrain': loss_terrain,
            'phys': loss_phys
        }
        return losses

    def pred(self, batch):
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain,
         controls_ts, controls,
         pose0,
         traj_ts, xs, xds, Rs, omegas) = batch
        # predict height maps
        img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        terrain = self.terrain_encoder(*img_inputs)

        # predict states
        states_pred = self.predicts_states(terrain, pose0, controls)

        return terrain, states_pred


class Points(ROUGH):
    def __init__(self, path, lss_cfg=None, dphys_cfg=DPhysConfig(), is_train=True):
        super(Points, self).__init__(path, lss_cfg, dphys_cfg=dphys_cfg, is_train=is_train)

    def get_sample(self, i):
        points = torch.as_tensor(position(self.get_cloud(i))).T
        control_ts, controls = self.get_controls(i)
        traj_ts, states = self.get_states_traj(i)
        xs, xds, Rs, omegas = states
        hm_geom = self.get_geom_height_map(i)
        hm_terrain = self.get_terrain_height_map(i)
        pose0 = torch.as_tensor(self.get_initial_pose_on_heightmap(i), dtype=torch.float32)
        return (points, hm_geom, hm_terrain,
                control_ts, controls,
                pose0,
                traj_ts, xs, xds, Rs, omegas)

class TrainerVoxelNet(TrainerCore):
        def __init__(self, dphys_cfg, lss_cfg, model='voxelnet', bsz=1, lr=1e-3, nepochs=1000,
                    pretrained_model_path=None, debug=False, vis=False, geom_weight=1.0, terrain_weight=1.0, phys_weight=0.1):
            super().__init__(dphys_cfg, lss_cfg, model, nepochs, debug, geom_weight, terrain_weight, phys_weight)

            # create dataloaders
            self.train_loader, self.val_loader = self.create_dataloaders(bsz=bsz, debug=debug, vis=vis, Data=Points)

            # load models: terrain encoder
            self.terrain_encoder = VoxelNet(grid_conf=self.lss_cfg['grid_conf'],
                                            outC=1).from_pretrained(pretrained_model_path)
            self.terrain_encoder.to(self.device)
            self.terrain_encoder.train()

            # define optimizer
            self.optimizer = torch.optim.Adam(self.terrain_encoder.parameters(), lr=lr)

        def compute_losses(self, batch):
            (points, hm_geom, hm_terrain,
             control_ts, controls,
             pose0,
             traj_ts, xs, xds, Rs, omegas) = batch
            # terrain encoder forward pass
            points_input = points
            terrain = self.terrain_encoder(points_input)

            # geometry loss: difference between predicted and ground truth height maps
            if self.geom_weight > 0:
                loss_geom = hm_loss(terrain['geom'], hm_geom[:, 0:1], hm_geom[:, 1:2])
                loss_geom += 0.02 * self.laplacian_loss(terrain['geom'])
            else:
                loss_geom = torch.tensor(0.0, device=self.device)

            # rigid / terrain height map loss
            if self.terrain_weight > 0:
                loss_terrain = hm_loss(terrain['terrain'], hm_terrain[:, 0:1], hm_terrain[:, 1:2])
                loss_terrain += 0.02 * self.laplacian_loss(terrain['terrain'])
            else:
                loss_terrain = torch.tensor(0.0, device=self.device)

            # physics loss: difference between predicted and ground truth states
            states_gt = [xs, xds, Rs, omegas]
            states_pred = self.predicts_states(terrain, pose0, controls)

            if self.phys_weight > 0:
                loss_phys = physics_loss(states_pred=states_pred, states_gt=states_gt,
                                              pred_ts=control_ts, gt_ts=traj_ts)
            else:
                loss_phys = torch.tensor(0.0, device=self.device)

            losses = {
                'geom': loss_geom,
                'terrain': loss_terrain,
                'phys': loss_phys
            }
            return losses

        def pred(self, batch):
            (points, hm_geom, hm_terrain,
             control_ts, controls,
             pose0,
             traj_ts, xs, xds, Rs, omegas) = batch

            # predict terrain
            terrain = self.terrain_encoder(points)

            # predict states
            states_pred = self.predicts_states(terrain, pose0, controls)

            return terrain, states_pred


class Fusion(ROUGH):
    def __init__(self, path, lss_cfg=None, dphys_cfg=DPhysConfig(), is_train=True):
        super(Fusion, self).__init__(path, lss_cfg, dphys_cfg=dphys_cfg, is_train=is_train)

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_images_data(i)
        points = torch.as_tensor(position(self.get_cloud(i))).T
        control_ts, controls = self.get_controls(i)
        traj_ts, states = self.get_states_traj(i)
        xs, xds, Rs, omegas = states
        hm_geom = self.get_geom_height_map(i)
        hm_terrain = self.get_terrain_height_map(i)
        pose0 = torch.as_tensor(self.get_initial_pose_on_heightmap(i), dtype=torch.float32)
        return (imgs, rots, trans, intrins, post_rots, post_trans,
                hm_geom, hm_terrain,
                control_ts, controls,
                pose0,
                traj_ts, xs, xds, Rs, omegas,
                points)

class TrainerBEVFusion(TrainerCore):

    def __init__(self, dphys_cfg, lss_cfg, model='bevfusion', bsz=1, lr=1e-3, nepochs=1000,
                 pretrained_model_path=None, debug=False, vis=False, geom_weight=1.0, terrain_weight=1.0, phys_weight=0.1):
        super().__init__(dphys_cfg, lss_cfg, model, nepochs, debug, geom_weight, terrain_weight, phys_weight)

        # create dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders(bsz=bsz, debug=debug, vis=vis, Data=Fusion)

        # load models: terrain encoder
        self.terrain_encoder = BEVFusion(grid_conf=self.lss_cfg['grid_conf'],
                                         data_aug_conf=self.lss_cfg['data_aug_conf']).from_pretrained(pretrained_model_path)
        self.terrain_encoder.to(self.device)
        self.terrain_encoder.train()

        # define optimizer
        self.optimizer = torch.optim.Adam(self.terrain_encoder.parameters(), lr=lr)

    def compute_losses(self, batch):
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain,
         control_ts, controls,
         pose0,
         traj_ts, xs, xds, Rs, omegas,
         points) = batch
        # terrain encoder forward pass
        img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        points_input = points
        terrain = self.terrain_encoder(img_inputs, points_input)

        # geometry loss: difference between predicted and ground truth height maps
        if self.geom_weight > 0:
            loss_geom = hm_loss(terrain['geom'], hm_geom[:, 0:1], hm_geom[:, 1:2])
            loss_geom += 0.02 * self.laplacian_loss(terrain['geom'])
        else:
            loss_geom = torch.tensor(0.0, device=self.device)

        # rigid / terrain height map loss
        if self.terrain_weight > 0:
            loss_terrain = hm_loss(terrain['terrain'], hm_terrain[:, 0:1], hm_terrain[:, 1:2])
            loss_terrain += 0.02 * self.laplacian_loss(terrain['terrain'])
        else:
            loss_terrain = torch.tensor(0.0, device=self.device)

        # physics loss: difference between predicted and ground truth states
        states_gt = [xs, xds, Rs, omegas]
        states_pred = self.predicts_states(terrain, pose0, controls)

        if self.phys_weight > 0:
            loss_phys = physics_loss(states_pred=states_pred, states_gt=states_gt,
                                     pred_ts=control_ts, gt_ts=traj_ts)
        else:
            loss_phys = torch.tensor(0.0, device=self.device)

        losses = {
            'geom': loss_geom,
            'terrain': loss_terrain,
            'phys': loss_phys
        }
        return losses

    def pred(self, batch):
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain,
         controls_ts, controls,
         pose0,
         traj_ts, xs, xds, Rs, omegas,
         points) = batch

        # predict height maps
        img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        terrain = self.terrain_encoder(img_inputs, points)

        # predict states
        states_pred = self.predicts_states(terrain, pose0, controls)

        return terrain, states_pred


def choose_trainer(model):
    if model == 'lss':
        return TrainerLSS
    elif model == 'bevfusion':
        return TrainerBEVFusion
    elif model == 'voxelnet':
        return TrainerVoxelNet
    else:
        raise ValueError(f'Invalid model: {model}. Supported models: lss, bevfusion, voxelnet')


def main():
    args = arg_parser()
    print(args)

    # load configs: DPhys and LSS (terrain encoder)
    dphys_cfg = DPhysConfig(robot=args.robot, grid_res=args.dphys_grid_res)
    dphys_cfg.traj_sim_time = args.traj_sim_time
    lss_config_path = args.lss_cfg_path
    assert os.path.isfile(lss_config_path), 'LSS config file %s does not exist' % lss_config_path
    lss_cfg = read_yaml(lss_config_path)

    # create trainer
    Trainer = choose_trainer(args.model)
    trainer = Trainer(model=args.model,
                      dphys_cfg=dphys_cfg, lss_cfg=lss_cfg,
                      bsz=args.bsz, nepochs=args.nepochs,
                      lr=args.lr,
                      pretrained_model_path=args.pretrained_model_path,
                      geom_weight=args.geom_weight,
                      terrain_weight=args.terrain_weight,
                      phys_weight=args.phys_weight,
                      debug=args.debug, vis=args.vis)
    # start training
    trainer.train()


if __name__ == '__main__':
    main()
