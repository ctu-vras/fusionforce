#!/usr/bin/env python3

import os
from copy import copy
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from PIL import Image as PILImage

from fusionforce.utils import read_yaml
from fusionforce.models.terrain_encoder.lss import LiftSplatShoot
from fusionforce.models.terrain_encoder.voxelnet import VoxelNet
from fusionforce.models.terrain_encoder.bevfusion import BEVFusion
from fusionforce.models.terrain_encoder.utils import get_image_augmentations, img_transform, normalize_img
from fusionforce.utils import set_device
from fusionforce.ros import terrain_to_gridmap_msg
from fusionforce.transformations import transform_cloud

import rclpy
import rclpy.time
from rclpy.executors import ExternalShutdownException
from rclpy.impl.logging_severity import LoggingSeverity
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, CameraInfo
from sensor_msgs.msg import PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from grid_map_msgs.msg import GridMap
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros


fusionforce_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../'))


class TerrainEncoder(Node):

    def __init__(self):
        super().__init__('terrain_encoder')
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('model', 'lss')  # options: 'lss', 'voxelnet', 'bevfusion'
        self.declare_parameter('lss_cfg_path', os.path.join(fusionforce_path, 'config/lss_cfg.yaml'))
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('fixed_frame', 'odom')
        self.declare_parameter('img_topics', ['/camera_front/image_color/compressed'])
        self.declare_parameter('camera_info_topics', ['/camera_front/image_color/camera_info'])
        self.declare_parameter('cloud_topic', '/points')
        self.declare_parameter('max_msgs_delay', 0.1)
        self.declare_parameter('max_age', 0.5)

        self.device = set_device(self.get_parameter('device').value)
        self._logger.set_level(LoggingSeverity.DEBUG)

        self.lss_cfg = read_yaml(self.get_parameter('lss_cfg_path').get_parameter_value().string_value)
        self.model = self.get_parameter('model').get_parameter_value().string_value
        self.terrain_encoder = self.load_terrain_encoder(model=self.model)

        self.robot_frame = self.get_parameter('robot_frame').get_parameter_value().string_value
        self.fixed_frame = self.get_parameter('fixed_frame').get_parameter_value().string_value

        self.img_topics = self.get_parameter('img_topics').get_parameter_value().string_array_value
        self.camera_info_topics = self.get_parameter('camera_info_topics').get_parameter_value().string_array_value
        assert len(self.img_topics) == len(self.camera_info_topics)
        self.cloud_topic = self.get_parameter('cloud_topic').get_parameter_value().string_value

        self.cv_bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.time.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.max_msgs_delay = self.get_parameter('max_msgs_delay').get_parameter_value().double_value
        self.max_age = self.get_parameter('max_age').get_parameter_value().double_value

        # grid map publisher
        self.gridmap_pub = self.create_publisher(GridMap, '/terrain/grid_map', 10)

    def load_terrain_encoder(self, model='lss'):
        weights = os.path.join(fusionforce_path, f'config/weights/{model}/val.pth')
        self._logger.info(f'Loading terrain encoder from {weights}')
        if not os.path.exists(weights):
            self._logger.error(f'Model weights file {weights} does not exist. Using random weights.')
        if model == 'lss':
            terrain_encoder = LiftSplatShoot(self.lss_cfg['grid_conf'],
                                             self.lss_cfg['data_aug_conf']).from_pretrained(weights)
        elif model == 'voxelnet':
            terrain_encoder = VoxelNet(self.lss_cfg['grid_conf']).from_pretrained(weights)
        elif model == 'bevfusion':
            terrain_encoder = BEVFusion(self.lss_cfg['grid_conf'],
                                        self.lss_cfg['data_aug_conf']).from_pretrained(weights)
        else:
            self._logger.error('Unknown model: %s' % model)
            raise (RuntimeError('Unknown model: %s' % model))
        terrain_encoder.to(self.device)
        terrain_encoder.eval()
        return terrain_encoder

    def spin(self):
        try:
            rclpy.spin(self)
        except (KeyboardInterrupt, ExternalShutdownException):
            self.get_logger().info('Keyboard interrupt, shutting down...')
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def start(self):
        # subscribe to topics with approximate time synchronization
        subs = []
        if self.model == 'lss':
            for topic in self.img_topics:
                self._logger.info('Subscribing to %s' % topic)
                subs.append(Subscriber(self, CompressedImage, topic))
            for topic in self.camera_info_topics:
                self._logger.info('Subscribing to %s' % topic)
                subs.append(Subscriber(self, CameraInfo, topic))
        elif self.model == 'voxelnet':
            self._logger.info('Subscribing to %s' % self.cloud_topic)
            subs.append(Subscriber(self, PointCloud2,self.cloud_topic))
        elif self.model == 'bevfusion':
            for topic in self.img_topics:
                self._logger.info('Subscribing to %s' % topic)
                subs.append(Subscriber(self, CompressedImage, topic))
            for topic in self.camera_info_topics:
                self._logger.info('Subscribing to %s' % topic)
                subs.append(Subscriber(self, CameraInfo, topic))
            self._logger.info('Subscribing to %s' % self.cloud_topic)
            subs.append(Subscriber(self, PointCloud2, self.cloud_topic))
        else:
            self._logger.error('Unknown model: %s' % self.model)
            raise (RuntimeError('Unknown model: %s' % self.model))
        sync = ApproximateTimeSynchronizer(subs, queue_size=1, slop=self.max_msgs_delay)
        sync.registerCallback(self.callback)

    def callback(self, *msgs):
        self._logger.debug('Received %d messages' % len(msgs))
        # if a message is stale, do not process it
        t_now = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec / 1e9
        t_msg = msgs[0].header.stamp.sec + msgs[0].header.stamp.nanosec / 1e9
        dt = abs(t_now - t_msg)
        if dt > self.max_age:
            self._logger.warning(f'Message is too old (time diff: {dt:.3f} > {self.max_age} s), skipping...')
        else:
            # process the messages
            self.proc(*msgs)

    def msgs_to_terrain(self, msgs):
        if self.model == 'lss':
            cam_inputs = self.cam_msgs_to_input(msgs)
            terrain = self.terrain_encoder(*cam_inputs)
        elif self.model == 'voxelnet':
            points_input = self.cloud_msg_to_input(msgs[-1])
            terrain = self.terrain_encoder(points_input)
        elif self.model == 'bevfusion':
            cam_inputs = self.cam_msgs_to_input(msgs[:-1])
            points_input = self.cloud_msg_to_input(msgs[-1])
            terrain = self.terrain_encoder(cam_inputs, points_input)
        else:
            raise RuntimeError(f'Unknown model {self.model}. Supported are lss, voxelnet, and bevfusion.')
        return terrain

    def cam_msgs_to_input(self, msgs):
        n = len(msgs)
        self._logger.debug('Received %d messages' % n)
        assert n % 2 == 0
        for i in range(n // 2):
            assert isinstance(msgs[i], CompressedImage), 'First %d messages must be CompressedImage' % (n // 2)
            assert isinstance(msgs[i + n // 2], CameraInfo), 'Last %d messages must be CameraInfo' % (n // 2)
            assert msgs[i].header.frame_id == msgs[i + n // 2].header.frame_id, \
                'Image and CameraInfo messages must have the same frame_id'
        img_msgs = msgs[:n // 2]
        info_msgs = msgs[n // 2:]
        cam_inputs = self.get_lss_inputs(img_msgs, info_msgs)
        cam_inputs = [i.to(self.device) for i in cam_inputs]
        return cam_inputs

    def cloud_msg_to_input(self, msg):
        assert isinstance(msg, PointCloud2)
        points = pc2.read_points_numpy(msg, field_names=['x', 'y', 'z'], skip_nans=False)

        # transform points to robot frame
        Tr = self.get_transform(from_frame=msg.header.frame_id, to_frame=self.robot_frame,
                                time=msg.header.stamp)
        points = transform_cloud(points, Tr)

        # convert points to gravity-aligned frame
        robot_pose = self.get_transform(from_frame=self.robot_frame, to_frame=self.fixed_frame)
        roll, pitch, yaw = Rotation.from_matrix(robot_pose[:3, :3]).as_euler('xyz')
        R = Rotation.from_euler('xyz', [roll, pitch, 0]).as_matrix()
        points = points @ R.T  # rotate points to align with gravity

        points_input = torch.as_tensor(points, dtype=torch.float32).to(self.device)
        points_input = points_input.T[None]  # (1, 3, N)
        self._logger.debug('Input point cloud shape: %s' % str(points_input.shape))
        return points_input

    @torch.inference_mode()
    def proc(self, *msgs):
        terrain = self.msgs_to_terrain(msgs)

        # publish terrain as a grid map
        stamp = msgs[0].header.stamp
        grid_msg = terrain_to_gridmap_msg(layers=[terrain['terrain'].squeeze().cpu().numpy()], layer_names=['terrain'],
                                          grid_res=self.lss_cfg['grid_conf']['xbound'][2])
        grid_msg.header.stamp = stamp
        grid_msg.header.frame_id = self.robot_frame
        self.gridmap_pub.publish(grid_msg)

    def get_transform(self, from_frame, to_frame, time=None):
        """Retrieve a transformation matrix between two frames using TF2."""
        if time is None:
            time = rclpy.time.Time()
        timeout = rclpy.time.Duration(seconds=1.0)
        try:
            tf = self.tf_buffer.lookup_transform(to_frame, from_frame,
                                                 time=time, timeout=timeout)
        except Exception as ex:
            tf = self.tf_buffer.lookup_transform(to_frame, from_frame,
                                                 time=rclpy.time.Time(), timeout=timeout)
            self._logger.warning(
                f"Could not find transform from {from_frame} to {to_frame} at time {time}, using latest available transform: {ex}"
            )
        # Convert TF2 transform message to a 4x4 transformation matrix
        translation = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
        qaut = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
        T = np.eye(4)
        R = Rotation.from_quat(qaut).as_matrix()
        T[:3, 3] = translation
        T[:3, :3] = R
        return T

    def get_cam_calib_from_info_msg(self, msg):
        """
        Get camera calibration parameters from CameraInfo message.
        :param msg: CameraInfo message
        :return: E - extrinsics (4x4),
                 K - intrinsics (3x3),
                 D - distortion coefficients (5,)
        """
        assert isinstance(msg, CameraInfo)

        # get camera extrinsics
        E = self.get_transform(from_frame=msg.header.frame_id,
                               to_frame=self.robot_frame,
                               time=msg.header.stamp)
        K = np.array(msg.k).reshape((3, 3))
        D = np.array(msg.d)

        return E, K, D

    def preprocess_img(self, img):
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        # preprocessing parameters (resize, crop)
        lss_cfg = copy(self.lss_cfg)
        lss_cfg['data_aug_conf']['H'], lss_cfg['data_aug_conf']['W'] = img.shape[:2]
        resize, resize_dims, crop, flip, rotate = get_image_augmentations(lss_cfg, is_train=False)
        img, post_rot2, post_tran2 = img_transform(PILImage.fromarray(img), post_rot, post_tran,
                                                   resize=resize,
                                                   resize_dims=resize_dims,
                                                   crop=crop,
                                                   flip=False,
                                                   rotate=0)
        # normalize image (subtraction of mean and division by std)
        img = normalize_img(img)

        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3, dtype=torch.float32)
        post_rot = torch.eye(3, dtype=torch.float32)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2

        return img, post_rot, post_tran

    def get_lss_inputs(self, img_msgs, info_msgs):
        """
        Get inputs for the LSS model from image and camera info messages.
        """
        assert len(img_msgs) == len(info_msgs)

        robot_pose = self.get_transform(from_frame=self.robot_frame,
                                        to_frame=self.fixed_frame,
                                        time=img_msgs[0].header.stamp)
        roll, pitch, yaw = Rotation.from_matrix(robot_pose[:3, :3]).as_euler('xyz')
        R = Rotation.from_euler('xyz', [roll, pitch, 0]).as_matrix()

        imgs = []
        post_rots = []
        post_trans = []
        intriniscs = []
        cams_to_robot = []
        for cam_i, (img_msg, info_msg) in enumerate(zip(img_msgs, info_msgs)):
            assert isinstance(img_msg, CompressedImage)
            assert isinstance(info_msg, CameraInfo)

            img = self.cv_bridge.compressed_imgmsg_to_cv2(img_msg)
            self._logger.debug('Input image shape: %s' % str(img.shape))
            # BGR to RGB
            img = img[..., ::-1]
            E, K, D = self.get_cam_calib_from_info_msg(info_msg)

            # extrinsics relative to gravity-aligned frame
            E[:3, :3] = R @ E[:3, :3]

            img, post_rot, post_tran = self.preprocess_img(img)
            imgs.append(img)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            intriniscs.append(K)
            cams_to_robot.append(E)

        # to arrays
        imgs = np.stack(imgs)
        post_rots = np.stack(post_rots)
        post_trans = np.stack(post_trans)
        intrins = np.stack(intriniscs)
        cams_to_robot = np.stack(cams_to_robot)
        rots, trans = cams_to_robot[:, :3, :3], cams_to_robot[:, :3, 3]
        self._logger.debug('Preprocessed image shape: %s' % str(imgs.shape))

        inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        inputs = [torch.as_tensor(i[np.newaxis], dtype=torch.float32) for i in inputs]

        return inputs


def main(args=None):
    rclpy.init(args=args)
    node = TerrainEncoder()
    node.start()
    node.spin()


if __name__ == '__main__':
    main()
