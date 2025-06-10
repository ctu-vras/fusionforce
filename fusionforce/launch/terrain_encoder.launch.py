import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import PythonExpression


def generate_launch_description():
    # Declare launch arguments
    img_topics_arg = DeclareLaunchArgument(
        'img_topics',
        default_value='/camera/image_raw/compressed',
        description='Comma-separated list of image topics'
    )
    camera_info_topics_arg = DeclareLaunchArgument(
        'camera_info_topics',
        default_value='/camera/image_raw/camera_info',
        description='Comma-separated list of camera info topics'
    )
    cloud_topic_arg = DeclareLaunchArgument('cloud_topic', default_value='/camera/depth/points',
                                            description='Point cloud topic to use for terrain encoding')
    model_arg = DeclareLaunchArgument(
        'model', default_value='voxelnet', description='Model type to use for terrain encoding'
    )
    lss_cfg_arg = DeclareLaunchArgument(
        'lss_cfg_path',
        default_value=os.path.join(get_package_share_directory('fusionforce'), 'config/lss_cfg.yaml'),
        description='Path to the LSS configuration file'
    )
    robot_frame_arg = DeclareLaunchArgument(
        'robot_frame', default_value='base_link', description='Robot base frame id'
    )
    fixed_frame_arg = DeclareLaunchArgument(
        'fixed_frame', default_value='odom', description='Fixed world frame id'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false', description='Use simulation time or not'
    )

    # Define the Node with parsed topic strings using PythonExpression
    node = Node(
        package='fusionforce',
        executable='terrain_encoder',
        name='terrain_encoder',
        output='screen',
        parameters=[{
            'img_topics': PythonExpression([
                '["" + topic.strip() for topic in "', LaunchConfiguration('img_topics'), '".split(",")]'
            ]),
            'camera_info_topics': PythonExpression([
                '["" + topic.strip() for topic in "', LaunchConfiguration('camera_info_topics'), '".split(",")]'
            ]),
            'cloud_topic': LaunchConfiguration('cloud_topic'),
            'model': LaunchConfiguration('model'),
            'lss_cfg_path': LaunchConfiguration('lss_cfg_path'),
            'robot_frame': LaunchConfiguration('robot_frame'),
            'fixed_frame': LaunchConfiguration('fixed_frame'),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }]
    )

    return LaunchDescription([
        img_topics_arg,
        camera_info_topics_arg,
        cloud_topic_arg,
        model_arg,
        lss_cfg_arg,
        robot_frame_arg,
        fixed_frame_arg,
        use_sim_time_arg,
        TimerAction(period=0.0, actions=[node])
    ])
