#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path

def callback(msg):
    global f
    pose_count = len(msg.poses)
    rospy.loginfo(f"成功接收到完整轨迹！共包含 {pose_count} 个位姿，正在写入文件...")
    
    # 直接遍历这唯一一次收到的完整数组，一次性全写进去
    for pose_stamped in msg.poses:
        t = pose_stamped.header.stamp.to_sec()
        x = pose_stamped.pose.position.x
        y = pose_stamped.pose.position.y
        z = pose_stamped.pose.position.z
        qx = pose_stamped.pose.orientation.x
        qy = pose_stamped.pose.orientation.y
        qz = pose_stamped.pose.orientation.z
        qw = pose_stamped.pose.orientation.w
        
# 写入标准的 20列 OpenVINS 格式，完全对齐 GT
        f.write("{:.5f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} 0.0000000000 0.0000000000 0.0000000000 0.0000000000 0.0000000000 0.0000000000 0.0000000000 0.0000000000 0.0000000000 0.0000000000 0.0000000000 0.0000000000\n".format(
                t, x, y, z, qx, qy, qz, qw))
        
    f.flush()
    rospy.loginfo("写入完成！准备自动退出...")
    # 核心优化：写完这波完整的直接通知 ROS 关闭当前节点
    rospy.signal_shutdown("One-shot recording finished.")

def clean_shutdown():
    global f
    if not f.closed:
        f.close()
        rospy.loginfo("文件已安全关闭。")

if __name__ == '__main__':
    rospy.init_node('path_to_txt_oneshot')
    rospy.on_shutdown(clean_shutdown)
    
    file_path = rospy.get_param('~output', '/tmp/traj_estimate.txt')
    topic_name = rospy.get_param('~topic', '/ov_msckf/pathimu')
    
    global f
    f = open(file_path, 'w')
    f.write("# timestamp(s) tx ty tz qx qy qz qw Pr11 Pr12 Pr13 Pr22 Pr23 Pr33 Pt11 Pt12 Pt13 Pt22 Pt23 Pt33\n")
   
    rospy.loginfo(f"等待接收最后一次完整 Path 话题: {topic_name}")
    # 订阅后只要触发一次 callback，就会跑完并关掉节点
    rospy.Subscriber(topic_name, Path, callback)
    rospy.spin()