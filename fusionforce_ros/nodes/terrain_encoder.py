#!/usr/bin/env python

import os
from copy import copy
from threading import RLock
import torch
import numpy as np
import rospy
from cv_bridge import CvBridge
from grid_map_msgs.msg import GridMap
from fusionforce.ros import height_map_to_gridmap_msg, cloud_msg_to_numpy
from fusionforce.utils import read_yaml, timing, load_calib
from fusionforce.models.terrain_encoder.lss import LiftSplatShoot
from fusionforce.models.terrain_encoder.voxelnet import VoxelNet
from fusionforce.models.terrain_encoder.bevfusion import BEVFusion
from fusionforce.models.terrain_encoder.utils import img_transform, normalize_img, sample_augmentation
from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2
from time import time
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros
from PIL import Image as PILImage
from ros_numpy import numpify
from scipy.spatial.transform import Rotation


lib_path = os.path.join(__file__, '..', '..', '..')


class TerrainEncoder:
    def __init__(self, lss_cfg: dict):
        self.lss_cfg = lss_cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rate = rospy.get_param('~rate', None)
        self.rate = rospy.Rate(rate) if rate is not None else None

        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')
        self.fixed_frame = rospy.get_param('~fixed_frame', 'odom')

        calib_path = rospy.get_param('~calib_path', '')
        self.calib = load_calib(calib_path)
        if self.calib is not None:
            rospy.loginfo('Loaded calibration from %s' % calib_path)

        img_topics = rospy.get_param('~img_topics', [])
        n_cams = rospy.get_param('~n_cams', None)
        self.img_topics = img_topics[:n_cams] if isinstance(n_cams, int) else img_topics
        camera_info_topics = rospy.get_param('~camera_info_topics', [])
        self.camera_info_topics = camera_info_topics[:n_cams] if isinstance(n_cams, int) else camera_info_topics
        assert len(self.img_topics) == len(self.camera_info_topics)
        self.cloud_topic = rospy.get_param('~cloud_topic', '/points')

        self.model = rospy.get_param('~model', 'lss')
        self.terrain_encoder = self.load_terrain_encoder(model=self.model)

        # cv bridge
        self.cv_bridge = CvBridge()
        # tf listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # grid map publisher
        self.gridmap_pub = rospy.Publisher('/grid_map/terrain', GridMap, queue_size=1)

        # lock for processing
        self.proc_lock = RLock()

        self.max_msgs_delay = rospy.get_param('~max_msgs_delay', 0.1)
        self.max_age = rospy.get_param('~max_age', 0.2)

    def load_terrain_encoder(self, model='lss'):
        weights = rospy.get_param('~weights', os.path.join(lib_path, f'config/weights/{model}/val.pth'))
        rospy.loginfo('Loading terrain endoder from %s' % weights)
        if not os.path.exists(weights):
            rospy.logerr('Model weights file %s does not exist. Using random weights.' % weights)
        if model == 'lss':
            terrain_encoder = LiftSplatShoot(self.lss_cfg['grid_conf'],
                                             self.lss_cfg['data_aug_conf']).from_pretrained(weights)
        elif model == 'voxelnet':
            terrain_encoder = VoxelNet(self.lss_cfg['grid_conf']).from_pretrained(weights)
        elif model == 'bevfusion':
            terrain_encoder = BEVFusion(self.lss_cfg['grid_conf'],
                                        self.lss_cfg['data_aug_conf']).from_pretrained(weights)
        else:
            rospy.logerr('Unknown model: %s' % model)
            raise (RuntimeError('Unknown model: %s' % model))
        terrain_encoder.to(self.device)
        terrain_encoder.eval()
        return terrain_encoder

    @staticmethod
    def spin():
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            pass

    def start(self):
        # subscribe to topics with approximate time synchronization
        subs = []
        if self.model == 'lss':
            for topic in self.img_topics:
                rospy.loginfo('Subscribing to %s' % topic)
                subs.append(Subscriber(topic, CompressedImage))
            for topic in self.camera_info_topics:
                rospy.loginfo('Subscribing to %s' % topic)
                subs.append(Subscriber(topic, CameraInfo))
        elif self.model == 'voxelnet':
            rospy.loginfo('Subscribing to %s' % self.cloud_topic)
            subs.append(Subscriber(self.cloud_topic, PointCloud2))
        elif self.model == 'bevfusion':
            for topic in self.img_topics:
                rospy.loginfo('Subscribing to %s' % topic)
                subs.append(Subscriber(topic, CompressedImage))
            for topic in self.camera_info_topics:
                rospy.loginfo('Subscribing to %s' % topic)
                subs.append(Subscriber(topic, CameraInfo))
            rospy.loginfo('Subscribing to %s' % self.cloud_topic)
            subs.append(Subscriber(self.cloud_topic, PointCloud2))
        else:
            rospy.logerr('Unknown model: %s' % self.model)
            raise (RuntimeError('Unknown model: %s' % self.model))
        sync = ApproximateTimeSynchronizer(subs, queue_size=1, slop=self.max_msgs_delay)
        sync.registerCallback(self.callback)

    def preprocess_img(self, img):
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        # preprocessing parameters (resize, crop)
        lss_cfg = copy(self.lss_cfg)
        lss_cfg['data_aug_conf']['H'], lss_cfg['data_aug_conf']['W'] = img.shape[:2]
        resize, resize_dims, crop, flip, rotate = sample_augmentation(lss_cfg, is_train=False)
        img, post_rot2, post_tran2 = img_transform(PILImage.fromarray(img), post_rot, post_tran,
                                                   resize=resize,
                                                   resize_dims=resize_dims,
                                                   crop=crop,
                                                   flip=False,
                                                   rotate=0)
        # normalize image (substraction of mean and division by std)
        img = normalize_img(img)

        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3, dtype=torch.float32)
        post_rot = torch.eye(3, dtype=torch.float32)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2

        return img, post_rot, post_tran

    def get_cam_calib_from_yaml(self, camera, robot_frame='base_link'):
        """
        Load camera calibration parameters from yaml file.
        :param calib_path: path to yaml file
        :return: E - extrinsics (4x4),
                 K - intrinsics (3x3),
                 D - distortion coefficients (5,)
        """
        assert self.calib is not None

        Tr_robot_cam = self.calib['transformations'][f'T_{robot_frame}__{camera}']['data']
        Tr_robot_cam = np.array(Tr_robot_cam, dtype=np.float32).reshape((4, 4))
        E = Tr_robot_cam
        K = np.array(self.calib[camera]['camera_matrix']['data'], dtype=np.float32).reshape((3, 3))
        D = np.array(self.calib[camera]['distortion_coefficients']['data'], dtype=np.float32)

        return E, K, D

    def get_transform(self, from_frame, to_frame):
        time = rospy.Time(0)
        timeout = rospy.Duration.from_sec(1.0)
        try:
            tf = self.tf_buffer.lookup_transform(to_frame, from_frame, time, timeout)
        except Exception as ex:
            rospy.logerr('Could not transform from %s to %s: %s.', from_frame, to_frame, ex)
            raise ex

        return np.array(numpify(tf.transform), dtype=np.float32).reshape((4, 4))

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
        E = self.get_transform(from_frame=msg.header.frame_id, to_frame=self.robot_frame)
        K = np.array(msg.K, dtype=np.float32).reshape((3, 3))
        D = np.array(msg.D, dtype=np.float32)

        return E, K, D

    def get_lss_inputs(self, img_msgs, info_msgs):
        """
        Get inputs for LSS model from image and camera info messages.
        """
        assert len(img_msgs) == len(info_msgs)

        robot_pose = self.get_transform(from_frame=self.robot_frame, to_frame=self.fixed_frame)
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
            rospy.logdebug('Input image shape: %s' % str(img.shape))
            # BGR to RGB
            img = img[..., ::-1]
            if self.calib is not None:
                rospy.logdebug('Using calibration from yaml file')
                cam_name = self.camera_info_topics[cam_i].split('/')[1]
                E, K, D = self.get_cam_calib_from_yaml(cam_name)
            else:
                rospy.logdebug('Using calibration from CameraInfo message')
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
        rospy.logdebug('Preprocessed image shape: %s' % str(imgs.shape))

        inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        inputs = [torch.as_tensor(i[np.newaxis], dtype=torch.float32) for i in inputs]

        return inputs

    def callback(self, *msgs):
        rospy.loginfo('Received %d messages' % len(msgs))
        # if message is stale do not process it
        dt = rospy.Time.now() - msgs[0].header.stamp
        if dt.to_sec() > self.max_age:
            rospy.logdebug(f'Stale image messages received ({dt.to_sec():.3f} > {self.max_age} [sec]), skipping')
            return

        with torch.no_grad():
            with self.proc_lock:
                self.proc(*msgs)

        if self.rate is not None:
            self.rate.sleep()

    def cam_msgs_to_terrain(self, msgs):
        t0 = time()
        n = len(msgs)
        assert n % 2 == 0
        for i in range(n // 2):
            assert isinstance(msgs[i], CompressedImage), 'First %d messages must be CompressedImage' % (n // 2)
            assert isinstance(msgs[i + n // 2], CameraInfo), 'Last %d messages must be CameraInfo' % (n // 2)
            assert msgs[i].header.frame_id == msgs[i + n // 2].header.frame_id, \
                'Image and CameraInfo messages must have the same frame_id'
        img_msgs = msgs[:n // 2]
        info_msgs = msgs[n // 2:]
        inputs = self.get_lss_inputs(img_msgs, info_msgs)
        inputs = [i.to(self.device) for i in inputs]
        t1 = time()
        rospy.logdebug('Preprocessing time: %.3f [sec]' % (t1 - t0))
        # model inference
        terrain = self.terrain_encoder(*inputs)
        rospy.loginfo('LSS inference time: %.3f [sec]' % (time() - t1))
        return terrain

    def cloud_msg_to_terrain(self, msg):
        assert isinstance(msg, PointCloud2)
        points = cloud_msg_to_numpy(msg)
        points_input = torch.as_tensor(points, dtype=torch.float32).to(self.device)
        points_input = points_input[None].T  # (3, N)
        terrain = self.terrain_encoder(points_input)
        return terrain

    def proc(self, *msgs):
        if self.model == 'lss':
            terrain = self.cam_msgs_to_terrain(msgs)
        elif self.model == 'voxelnet':
            terrain = self.cloud_msg_to_terrain(msgs[0])
        elif self.model == 'bevfusion':
            raise NotImplementedError
        else:
            raise RuntimeError(f'Unknown model {self.model}. Supported are lss, voxelnet, and bevfusion.')

        # publish height map as grid map
        height_terrain, friction = terrain['terrain'], terrain['friction']
        rospy.loginfo('Predicted height map shape: %s' % str(height_terrain.shape))
        stamp = msgs[0].header.stamp
        height = height_terrain.squeeze().cpu().numpy()
        grid_msg = height_map_to_gridmap_msg(height, grid_res=self.lss_cfg['grid_conf']['xbound'][2],
                                             xyz=np.array([0., 0., 0.]), q=np.array([0., 0., 0., 1.]))
        grid_msg.info.header.stamp = stamp
        grid_msg.info.header.frame_id = self.robot_frame
        self.gridmap_pub.publish(grid_msg)


def main():
    rospy.init_node('terrain_encoder', anonymous=True, log_level=rospy.DEBUG)

    # load configs
    lss_config_path = rospy.get_param('~lss_config_path', os.path.join(lib_path, 'config/lss_cfg.yaml'))
    assert os.path.isfile(lss_config_path), 'LSS config file %s does not exist' % lss_config_path
    lss_cfg = read_yaml(lss_config_path)

    node = TerrainEncoder(lss_cfg=lss_cfg)
    node.start()
    node.spin()


if __name__ == '__main__':
    main()
