# FusionForce: End-to-end Differentiable Neural-Symbolic Layer for Trajectory Prediction

[![Arxiv](http://img.shields.io/badge/paper-arxiv-critical.svg?style=plastic)](https://arxiv.org/abs/2502.10156)
[![Data](http://img.shields.io/badge/data-ROUGH-blue.svg?style=plastic)](https://drive.google.com/drive/folders/1nli-4YExqcBhl0mPNRUjSiNecX4yIcme?usp=sharing)
[![PhD Thesis](http://img.shields.io/badge/PhD-Thesis-green.svg?style=plastic)](https://www.overleaf.com/read/cpkjnzmqzycd#9033bb)

## Abstract

We propose end-to-end differentiable model that predicts robot trajectories on rough offroad terrain from camera images
and/or lidar point clouds. The model integrates a learnable component that predicts robot-terrain interaction forces with a
neural-symbolic layer that enforces the laws of classical mechanics and consequently improves generalization on out-of-
distribution data. The neural-symbolic layer includes a differentiable physics engine that computes the robot’s trajectory
by querying these forces at the points of contact with the terrain. As the proposed architecture comprises substantial
geometrical and physics priors, the resulting model can also be seen as a learnable physics engine conditioned on
real sensor data that delivers $10^4$ trajectories per second. We argue and empirically demonstrate that this architecture
reduces the sim-to-real gap and mitigates out-of-distribution sensitivity. The differentiability, in conjunction with the rapid
simulation speed, makes the model well-suited for various applications including model predictive control, trajectory
shooting, supervised and reinforcement learning, or SLAM.

## Pipeline

<img src="./fusionforce/docs/imgs/fusionforce.png" alt="Terrain Encoder" width="1000"/>

- The **FusionForce Predictor** estimates rich features in image and voxel domains.
- Both **camera and lidar features** are vertically projected on the ground plane.
- Multi-headed terrain encoder-decoder is used to predict terrain properties.
- **Neuro Symbolic Physics Engine** estimate forces at robot-terrain contacts.
- Resulting robot trajectory is integrated from these forces.

Learning employs three losses:
- trajectory loss, $L_{\tau}$, which measures the distance between the predicted and real trajectory;
- geometrical loss, $L_{g}$,  which measures the distance between the predicted
geometrical heightmap and lidar-estimated heightmap;
- terrain loss, $L_{t}$, which enforces rigid terrain on rigid semantic classes
revealed through image foundation model [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once).

### Terrain Encoder with unified BEV fusion

<img src="./fusionforce/docs/imgs/terrain_encoder.jpg" alt="Terrain Encoder" width="1000"/>

- The depth distribution for each frustum ray is created by "lifting" into 3D the calibrated camera features, [LSS](https://github.com/nv-tlabs/lift-splat-shoot).
- The depth distribution is then projected onto the ground plane to create a 2D image BEV features.
- The voxelized featured are computed from a 3D point cloud, [VoxelNet](https://arxiv.org/abs/1711.06396).
- The voxelized features are projected onto the ground plane to create a 2D lidar BEV features.
- The camera and lidar BEV maps are fused into a single BEV features using an encoder-decoder architecture.
- The fused BEV features is then used to predict the terrain properties, such as elevation and friction.

### Differentiable Physics Engine

<img src="./fusionforce/docs/imgs/dphysics.jpg" alt="Differentiable Physics Engine" width="1000"/>

The differentiability of the developed physics engine allows for the
end-to-end training of the model.
For example, it allows to learn the terrain properties from the robot's trajectories.

## Autonomous Navigation

<img src="./fusionforce/docs/imgs/navigation.png" alt="Autonomous Navigation" width="1000"/>

- The **GPU parallelization** of the physics engine allows the module to be used for real-time trajectory shooting.
- The **cost of the trajectories** is computed from the predicted robot-terrain interaction forces.
- The path to follow is selected based on the **cost of the trajectories and the distance to a waypoint**.

## Installation
The package is organized as a
[ROS 2](https://docs.ros.org/) package and can be installed using the following commands:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone git@github.com:ctu-vras/fusionforce.git
pip install -r fusionforce/config/singularity/requirements.txt

cd ~/ros2_ws/
rosdep install --from-path src --ignore-src -r
colcon build --symlink-install --packages-select fusionforce
```

## Usage

### Physics Engine

To test the Physics Engine, one can run the following command:

```bash
python fusionforce/scripts/run_physics_engine.py
```
The ROS2 node can be launched with the following command:

```bash
ros2 launch fusionforce physics_engine.launch.py
```
Topics:
- input: `/terrain/grid_map` - the terrain [GridMap](https://github.com/ANYbotics/grid_map) message with the terrain properties (elevation, friction, etc.).

- output: `/sampled_paths` - the predicted robot trajectories as a [MarkerArray](https://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/MarkerArray.html) message.
- output: `/path_costs` - the costs of the predicted trajectories as a [Float32MultiArray](https://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32MultiArray.html) message.

Parameters:
- `num_robots`: number of robots (trajectories) to simulate,
- `traj_sim_time`: simulation time for each trajectory in seconds,
- `gridmap_layer`: name of the elevation layer in the GridMap message,
- `max_age`: maximum age of the terrain map in seconds to be processed.

### FusionForce Predictor (Terrain Encoder)

The ROS2 node for the FusionForce Predictor can be launched with the following command:

```bash
ros2 launch fusionforce terrain_encoder.launch.py terrain_encoder:=<model> img_topics:=[<img_topic1>, .. , <img_topicN>] camera_info_topics:=[<info_topic1>, .. , <info_topicN>] cloud_topic:=<cloud_topic>
```

where `<model>` is one of the available models: `voxelnet`, `lss`, or `bevfusion`.

Topics:
- input: a list of `img_topics` as a [sensor_msgs/CompressedImage](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CompressedImage.html) messages (required for `lss`, and `bevfusion`),
- input: a list of `camera_info_topics` as a [sensor_msgs/CameraInfo](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html) messages (required for `lss` and `bevfusion`),
- input: `cloud_topic` as a [sensor_msgs/PointCloud2](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointCloud2.html) message (required for `voxelnet` and `bevfusion`).

- output: `/terrain/grid_map` as a [GridMap](https://github.com/ANYbotics/grid_map) message with the estimated terrain properties.

Parameters:
- `model`: the model to use for the terrain encoder, one of `voxelnet`, `lss`, or `bevfusion`,
- `robot_frame`: the frame of the robot, e.g., `base_link`,
- `fixed_frame`: the fixed frame for the gravity alignment, e.g., `map`,
- `max_msgs_delay`: maximum delay in seconds for the input messages to be processed,
- `max_age`: maximum age of the sensor input in seconds to be processed,

### FusionForce (Terrain Encoder + Physics Engine)

The ROS2 node for the FusionForce Predictor with the Physics Engine can be launched with the following command:

```bash
ros2 launch fusionforce fusionforce.launch.py terrain_encoder:=<model> img_topics:=[<img_topic1>, .. , <img_topicN>] camera_info_topics:=[<info_topic1>, .. , <info_topicN>] cloud_topic:=<cloud_topic>
```
The module combines the FusionForce Predictor and the Physics Engine into a single node.
It has the same input as the FusionForce Predictor, but also outputs the predicted robot trajectories and their costs,
as the Physics Engine does.

<img src="./fusionforce/docs/imgs/prediction.png" alt="FusionForce Node" width="1000"/>

The FusionForce prediction example: supporting terrain elavation projected to the robot's camera frames.

## Citation

Consider citing the paper if you find the work relevant to your research:

```bibtex
@article{agishev2025fusionforce,
  title={FusionForce: End-to-end Differentiable Neural-Symbolic Layer for Trajectory Prediction},
  author={Agishev, Ruslan and Zimmermann, Karel},
  journal={arXiv preprint arXiv:2502.10156},
  year={2025},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2502.10156},
}
```
