#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def callback(msg):
    bridge = CvBridge()
    try:
        # 1. 强制将 encoding 视为 bgr8 读取 (假设它是 BGR)
        # 如果 rosbag 里写的是 "8UC3"，cv_bridge 直接读取会得到原始数据
        # 我们这里先把它当作 bgr8 处理
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
        # 2. 如果是 3 通道，转为单通道灰度图
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = cv_image

        # 3. 发布 encoding 为 "mono8" 的新消息
        new_msg = bridge.cv2_to_imgmsg(gray_image, encoding="mono8")
        new_msg.header = msg.header # 保持时间戳一致
        pub.publish(new_msg)
        
    except Exception as e:
        rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node('thermal_republisher')
    
    # 订阅原始话题
    sub = rospy.Subscriber('/thermal_image_raw', Image, callback)
    
    # 发布给 OpenVINS 的新话题
    pub = rospy.Publisher('/thermal_mono', Image, queue_size=10)
    
    print("Republishing /thermal_image_raw")
    rospy.spin()