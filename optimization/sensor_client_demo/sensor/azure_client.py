#!/usr/bin/env python

import numpy as np
import cv2
from skimage.transform import resize

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge

"""
Azure RGBD ROS Node
"""
class AzureRGBDFetchNode:
    def __init__(self, depth_topic='/depth_to_rgb/hw_registered/image_rect_raw', 
            rgb_topic='/rgb/image_rect_color', CameraInfo_topic='/rgb/camera_info',
            use_depth=True, use_rgb=True):

        self.use_depth = use_depth
        self.use_rgb = use_rgb

        self.br = CvBridge()

        self.Image_depth_buff = None
        rospy.Subscriber(depth_topic, Image, self.depth_callback)
        rospy.loginfo('ROS subcriping to {}'.format(depth_topic))

        self.Image_rgb_buff = None
        rospy.Subscriber(rgb_topic, Image, self.rgb_callback)
        rospy.loginfo('ROS subcriping to {}'.format(rgb_topic))

        self.CamInfo_buff = None
        self.is_updated_CI = False
        rospy.Subscriber(CameraInfo_topic, CameraInfo, self.CInfo_callback)
        rospy.loginfo('ROS subcriping to {}'.format(CameraInfo_topic))

    def depth_callback(self, msg):
        self.Image_depth_buff = msg

    def rgb_callback(self, msg):
        self.Image_rgb_buff = msg

    def CInfo_callback(self, msg):
        """
            https://github.com/code-iai/pico_flexx_driver/blob/master/src/pico_flexx_driver.cpp#L906

            std_msgs/Header header
            uint32 height
            uint32 width
            string distortion_model
            float64[] D
            float64[9] K
            float64[9] R
            float64[12] P
            uint32 binning_x
            uint32 binning_y
            sensor_msgs/RegionOfInterest roi
        """
        self.CamInfo_buff = msg
    
    def fetch_image(self, inpaint_depth = True):
        rgb_image, depth_image = None, None
        output_shape = 300
        
        ## Camera Info
        if self.CamInfo_buff is None: return None, None
        else: cam_info = self.CamInfo_buff
        
        ## RGB Image
        if self.use_rgb and self.Image_rgb_buff is None: return None, None
        elif self.use_rgb:
            rgb_image = self.br.imgmsg_to_cv2(self.Image_rgb_buff)
            # alpha = 1.0 # Contrast control
            # beta = 2 # Brightness control
            # rgb_image = cv2.convertScaleAbs(rgb_image, alpha=alpha, beta=beta)

            height, width, _ = rgb_image.shape # (height 720, width 1080)
            rgb_image = rgb_image[:, (width - height)//2 : (width - height)//2 + height, :]
            mask = (rgb_image[:,:,3] == 0).astype(np.uint8)
            rgb_image = self.rgb_inpaint(rgb_image[:,:,:-1], mask)
            np.set_printoptions(threshold=np.inf)
            rgb_image = resize(rgb_image, output_shape=(output_shape,output_shape), 
                preserve_range=True).astype(rgb_image.dtype)

        ## Depth Image
        if self.use_depth and self.Image_depth_buff is None: return None, None
        elif self.use_depth:
            depth_image = self.br.imgmsg_to_cv2(self.Image_depth_buff, desired_encoding="passthrough")
            depth_image = np.array(depth_image, dtype=np.float32)/1000.0 # MM to M 
            height, width = depth_image.shape # (height 720, width 1080)
            depth_image = depth_image[:, (width - height)//2 : (width - height)//2 + height]
            
            if inpaint_depth: depth_image = self.depth_inpaint(depth_image)
            depth_image = resize(depth_image, output_shape=(output_shape,output_shape), 
                preserve_range=True).astype(depth_image.dtype)
        
        rospy.loginfo('Fetch depth_image size: {}'.format(depth_image.shape))
        # if depth_image is not None:
        #     plot = plt.matshow(depth_image, cmap='jet')
        #     plt.colorbar(plot)
        #     plt.savefig("/home/smarnlab/catkin_ws/src/gripper_sensor/gripper_cameras/depth_image.png")
        #     plt.close()
        
        zoom_ratio = float(output_shape) / float(height)
        fx_0, fy_0, cx_0, cy_0 = cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5]
        cam_info_update = {
            'fx': fx_0 * zoom_ratio, 'fy':fy_0 * zoom_ratio,
            'cx': (cx_0 - (width - height)//2) * zoom_ratio, 'cy':cy_0 * zoom_ratio
        }

        rgb_image = rgb_image[...,[2,1,0]].copy()
        if self.use_depth and not self.use_rgb:
            image = np.expand_dims(depth_image, axis=0)
        elif self.use_depth and self.use_rgb:
            image = np.concatenate((np.expand_dims(depth_image, axis=0), \
                np.transpose(rgb_image, (2, 0, 1))), axis=0)
        else: raise KeyError("Not implement.")
        print('image.shape', image.shape)
        return image, cam_info_update

    def fetch_ori_image(self, inpaint_depth = True):
        rgb_image, depth_image = None, None
        
        ## Camera Info
        if self.CamInfo_buff is None: return None, None
        else: cam_info = self.CamInfo_buff
        
        ## RGB Image
        if self.use_rgb and self.Image_rgb_buff is None: return None, None
        elif self.use_rgb:
            rgb_image = self.br.imgmsg_to_cv2(self.Image_rgb_buff)

            mask = (rgb_image[:,:,3] == 0).astype(np.uint8)
            rgb_image = self.rgb_inpaint(rgb_image[:,:,:-1], mask)

        ## Depth Image
        if self.use_depth and self.Image_depth_buff is None: return None, None
        elif self.use_depth:
            depth_image = self.br.imgmsg_to_cv2(self.Image_depth_buff, desired_encoding="passthrough")
            depth_image = np.array(depth_image, dtype=np.float32)/1000.0 # MM to M 
  
            if inpaint_depth: depth_image = self.depth_inpaint(depth_image)
        
        rospy.loginfo('Rgb, Depth_image size: {}, {}'.format(rgb_image.shape, depth_image.shape))

        fx_0, fy_0, cx_0, cy_0 = cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5]
        cam_info_update = {
            'fx': fx_0, 'fy':fy_0,
            'cx': cx_0, 'cy':cy_0
        }
        
        # if self.use_depth and not self.use_rgb:
        #     image = np.expand_dims(depth_image, axis=0)
        # elif self.use_depth and self.use_rgb:
        #     image = np.concatenate((np.expand_dims(depth_image, axis=0), \
        #         np.transpose(rgb_image, (2, 0, 1))), axis=0)
        # else: raise KeyError("Not implement.")
        # print('image.shape', image.shape)
        
        return (depth_image, rgb_image), cam_info_update
        
    @staticmethod
    def rgb_inpaint(image_o, mask):
        """
        Inpaint missing values in depth image.
        :param missing_value: Value to fill in teh depth image.
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        
        # mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        # image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        # image = cv2.inpaint(image, mask, 2, cv2.INPAINT_NS) # repair with fluid alg. radius 1
        # image = image[1:-1, 1:-1] # cut the 1 pixel boarder
        image = image_o.copy()
        patch_bias = 15
        image[:patch_bias, :, :] = image[patch_bias:patch_bias+patch_bias, :, :]
        image[-patch_bias:, :, :] = image[-(patch_bias+patch_bias): -patch_bias, :, :]
        return image

    @staticmethod
    def depth_inpaint(image, missing_value=0):
        """
        Inpaint missing values in depth image.
        :param missing_value: Value to fill in teh depth image.
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (image == missing_value).astype(np.uint8)
        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        imax, imin = np.abs(image).max(), np.abs(image).min()
        irange = imax - imin
        image = ((image- imin) / irange * 255.0).astype(np.uint8)  # Has be float32, 64 not supported. get -1:1
        image = cv2.inpaint(image, mask, 2, cv2.INPAINT_NS) # repair with fluid alg. radius 1
        # Back to original size and value range.
        image = image[1:-1, 1:-1] # cut the 1 pixel boarder
        image = image.astype(np.float32) / 255.0 * irange + imin
        return image

    @staticmethod
    def Reconstruct_XYZ(depth_value, point, intrinsics_info):
        """
        Convinient class 
        the first index of point is u, along x direction of image, or width
        Get X, Y, Z coordinate of a pixel on depth image.
        By: reconstructed from fxfypxpy
        where depth image is the z axis value
        """
        u, v = point[0], point[1] # in width, height
        cx, cy = intrinsics_info['cx'], intrinsics_info['cy']
        fx_inv, fy_inv = 1/intrinsics_info['fx'], 1/intrinsics_info['fy']
        _z = depth_value
        _x = _z * ((u - cx) * fx_inv)
        _y = _z * ((v - cy) * fy_inv)
        xyz = [_x, _y, _z]
        return xyz
