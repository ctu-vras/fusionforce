# Installation

Install Python dependencies:
```commandline
pip install -r ../docker/requirements.txt
```

## ROS Integration
Please clone the ROS package, install its dependencies, and build the workspace:
```commandline
mkdir -p ~/traversability_ws/src/
cd ~/traversability_ws/src/
git clone https://github.com/ctu-vras/fusionforce.git

cd ~/traversability_ws/
catkin init
catkin config --extend /opt/ros/$ROS_DISTRO/
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
rosdep install --from-paths src --ignore-src -r -y
catkin build
```


## Docker

Please, install
[Docker](https://docs.docker.com/engine/install/ubuntu/)
and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

To build the image please run:
```commandline
cd ../docker/
make build
```

## Singularity

It is possible to run the fusionforce package in a [Singularity](https://sylabs.io/singularity/) container.
To build the image:
```commandline
cd ../singularity/
./build.sh
```
