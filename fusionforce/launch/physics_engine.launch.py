from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='fusionforce',
            executable='physics_engine',
            name='physics_engine_node',
            output='screen',
            parameters=[
                {
                    'gridmap_topic': '/terrain/grid_map',
                    'gridmap_layer': 'terrain',
                    'robot_frame': 'base_link',
                }
            ]
        )
    ])
