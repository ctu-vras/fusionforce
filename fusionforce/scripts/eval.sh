#!/bin/bash

echo "Source ROS workspace..."

SEQ=val
BATCH_SIZE=1
TERRAIN_ENCODERS=(lss voxelnet bevfusion)
TRAJ_PREDICTORS=(dphysics)
VIS=True

for TERRAIN_ENCODER in "${TERRAIN_ENCODERS[@]}"
do
  for TRAJ_PREDICTOR in "${TRAJ_PREDICTORS[@]}"
  do
    WEIGHTS=$HOME/workspaces/ros1/traversability_ws/src/fusionforce/fusionforce/config/weights/${TERRAIN_ENCODER}/val.pth
    echo "Evaluating terrain encoder ${TERRAIN_ENCODER} with trajectory predictor ${TRAJ_PREDICTOR}..."
    ./eval.py --terrain_encoder ${TERRAIN_ENCODER} \
              --terrain_encoder_path ${WEIGHTS} \
              --traj_predictor ${TRAJ_PREDICTOR} \
              --batch_size ${BATCH_SIZE} \
              --seq ${SEQ} \
              --vis ${VIS}
  done
done

echo "Done evaluating."
