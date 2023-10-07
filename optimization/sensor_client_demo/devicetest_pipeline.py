#!/usr/bin/env python

###
# Test RGBD node with inference module
###

from client.sockect_client import ImageStrServerClient
from sensor.azure_client import AzureRGBDFetchNode

import rospy
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped


"""
Main function
"""
if __name__=='__main__':
    rospy.init_node('grasp_pipeline', anonymous=True)
    
    ### Initial class
    rgbd_node = AzureRGBDFetchNode(use_depth=True, use_rgb=True)
    client = ImageStrServerClient('CLIENT')
    
    ### Start grasp pipeline
    loop_rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        fed_image, fed_cam_info = rgbd_node.fetch_image()
        
        if fed_image is None:
            rospy.loginfo('Awaiting image from depth device ...')
        else:
            # Obtain detection result
            client.Send(fed_image, 'TestImage{}'.format(0))
            grasp, strs = client.Get(8 * 8, 10, shape=(8))
            [p_x, p_y, wid, ang, x1, x2, y1, y2] = grasp

            # Gain grasp points, argument 1 and 2 should be revert
            in_x, in_y = p_x, p_y
            xyz = rgbd_node.Reconstruct_XYZ(fed_image[0, int(in_x), int(in_y)], (in_y, in_x), fed_cam_info)
            print('p_x, p_y', p_x, p_y, '\nxyz ang, wid:', xyz, ang, wid)

        loop_rate.sleep()