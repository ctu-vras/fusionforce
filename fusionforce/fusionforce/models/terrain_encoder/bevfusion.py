import torch
import torch.nn as nn
from .voxelnet import LidarNet
from .lss import LiftSplatShoot, BevEncode


class BEVFusion(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, out_channels=1):
        super().__init__()
        self.lss = LiftSplatShoot(grid_conf, data_aug_conf)
        del self.lss.bevencode  # Remove the original bevencode to replace it with a new one
        self.lidar_net = LidarNet(grid_conf=grid_conf, out_channels=64)
        # Initialize BevEncode with the correct input channels: # 2 * 64 (from both camera and LiDAR)
        self.bevencode = BevEncode(in_channels=2*64, out_channels=out_channels)

    def forward(self, img_inputs, cloud_input):
        # Get BEV features from camera inputs
        cam_feat_bev = self.lss.get_voxels(*img_inputs)  # Shape (B, Z, X, Y)

        # Get BEV features from LiDAR inputs
        lidar_feat_bev = self.lidar_net(cloud_input)  # Shape (B, Z, X, Y)

        # Concatenate the two BEV features
        feat_bev = torch.cat([cam_feat_bev, lidar_feat_bev], dim=1)  # Shape (B, 2xZ, X, Y)

        # Encode the concatenated BEV features
        feat_bev = self.bevencode.backbone(feat_bev)  # Shape (B, 2xZ, X, Y)

        # Apply the up-convolutional heads
        out = self.bevencode.heads(feat_bev)  # Shape (B, outC, X, Y)

        return out

    def from_pretrained(self, modelf):
        if not modelf:
            return self
        print(f'Loading pretrained {self.__class__.__name__} model from', modelf)
        # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
        # model dict
        model_dict = self.state_dict()
        # load pretrained model
        pretrained_dict = torch.load(modelf)
        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # update model dict with pretrained model
        model_dict.update(pretrained_dict)
        # load the updated model dict into the current model
        self.load_state_dict(model_dict)
        return self
