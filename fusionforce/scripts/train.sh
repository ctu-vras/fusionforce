#!/bin/bash

MODEL=bevfusion  # lss, voxelnet, bevfusion
ROBOT=marv
DEBUG=True
VIS=True
BSZ=2  # 24, 24, 4
WEIGHTS=$HOME/workspaces/ros1/traversability_ws/src/fusionforce/fusionforce/config/weights/${MODEL}/val.pth

./train.py --bsz $BSZ --nepochs 1000 --lr 1e-4 \
           --debug $DEBUG --vis $VIS \
           --geom_weight 1.0 --terrain_weight 2.0 --phys_weight 1.0 \
           --traj_sim_time 5.0 \
           --robot $ROBOT \
           --model $MODEL \
           --pretrained_model_path ${WEIGHTS}
