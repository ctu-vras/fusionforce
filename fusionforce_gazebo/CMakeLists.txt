cmake_minimum_required(VERSION 3.0.2)
project(fusionforce_gazebo)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
)

catkin_package(
  CATKIN_DEPENDS
  gazebo_ros
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)

install(DIRECTORY worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds
)

install(
    PROGRAMS
        nodes/robot_gt_pose_to_odom
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
