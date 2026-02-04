#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较两个VIO性能CSV文件的速度差异
确保时间戳对齐后进行比较
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='比较两个VIO性能CSV文件的速度差异')
    parser.add_argument('--file1', type=str, default='/root/sqrt_vins_ws/logs/vio_performance.csv',
                        help='第一个CSV文件路径')
    parser.add_argument('--file2', type=str, default='/root/catkin_ws/src/open_vins/logs/vio_performance.csv',
                        help='第二个CSV文件路径')
    parser.add_argument('--save', action='store_true', help='保存图表而不显示')
    parser.add_argument('--output_dir', type=str, default='/root/',
                        help='图表保存目录')
    parser.add_argument('--time_threshold', type=float, default=0.01, 
                        help='时间对齐阈值（秒）')
    return parser.parse_args()



def load_data(file_path):
    """加载CSV数据"""
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None
    
    try:
        data = pd.read_csv(file_path)
        print(f"成功加载 {file_path}：{len(data)} 行，{len(data.columns)} 列")
        return data
    except Exception as e:
        print(f"加载 {file_path} 时出错：{e}")
        return None



def align_data(data1, data2, time_threshold=0.01):
    """根据时间戳对齐两个数据集
    
    参数:
        data1: 第一个数据集
        data2: 第二个数据集
        time_threshold: 时间对齐阈值（秒）
        
    返回:
        aligned_data1: 对齐后的第一个数据集
        aligned_data2: 对齐后的第二个数据集
    """
    # 确保时间戳是数值类型
    data1['timestamp'] = pd.to_numeric(data1['timestamp'])
    data2['timestamp'] = pd.to_numeric(data2['timestamp'])
    
    # 对两个数据集按时间戳排序
    data1_sorted = data1.sort_values('timestamp').reset_index(drop=True)
    data2_sorted = data2.sort_values('timestamp').reset_index(drop=True)
    
    # 初始化对齐后的数据集
    aligned_data1 = []
    aligned_data2 = []
    
    # 使用双指针法进行对齐
    i = 0
    j = 0
    
    while i < len(data1_sorted) and j < len(data2_sorted):
        time1 = data1_sorted.loc[i, 'timestamp']
        time2 = data2_sorted.loc[j, 'timestamp']
        
        # 计算时间差
        time_diff = abs(time1 - time2)
        
        if time_diff <= time_threshold:
            # 时间差在阈值内，将两个数据点都加入对齐后的数据集
            aligned_data1.append(data1_sorted.loc[i].copy())
            aligned_data2.append(data2_sorted.loc[j].copy())
            i += 1
            j += 1
        elif time1 < time2:
            # 第一个数据集的时间戳更小，移动第一个数据集的指针
            i += 1
        else:
            # 第二个数据集的时间戳更小，移动第二个数据集的指针
            j += 1
    
    # 转换为DataFrame
    aligned_data1 = pd.DataFrame(aligned_data1)
    aligned_data2 = pd.DataFrame(aligned_data2)
    
    print(f"对齐后的数据点数量：{len(aligned_data1)}个")
    
    return aligned_data1, aligned_data2



def plot_vel_diff_comparison(data1, data2, file1_name, file2_name, args):
    """绘制速度差异对比图"""
    # 计算相对时间（从第一个时间戳开始）
    time = data1['timestamp'] - data1['timestamp'].iloc[0]
    
    # 创建三个图表，分别对应x, y, z方向
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # X方向速度差异对比
    axes[0].plot(time, data1['vel_diff_x'], '-', linewidth=1, color='blue', label='vel_diff_x_sqrt')
    axes[0].plot(time, data2['vel_diff_x'], '-', linewidth=1, color='red', label='vel_diff_x_ov')
    axes[0].set_title('Velocity Difference Comparison - X Direction', fontsize=14)
    axes[0].set_ylabel('Velocity Difference (m/s)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True)
    
    # Y方向速度差异对比
    axes[1].plot(time, data1['vel_diff_y'], '-', linewidth=1, color='blue', label='vel_diff_y_sqrt')
    axes[1].plot(time, data2['vel_diff_y'], '-', linewidth=1, color='red', label='vel_diff_y_ov')
    axes[1].set_title('Velocity Difference Comparison - Y Direction', fontsize=14)
    axes[1].set_ylabel('Velocity Difference (m/s)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True)
    
    # Z方向速度差异对比
    axes[2].plot(time, data1['vel_diff_z'], '-', linewidth=1, color='blue', label='vel_diff_z_sqrt')
    axes[2].plot(time, data2['vel_diff_z'], '-', linewidth=1, color='red', label='vel_diff_z_ov')
    axes[2].set_title('Velocity Difference Comparison - Z Direction', fontsize=14)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Velocity Difference (m/s)', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if args.save:
        # 确保输出目录存在
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # 保存图表
        plt.savefig(os.path.join(args.output_dir, 'vel_diff_comparison.png'), dpi=300, bbox_inches='tight')
        print("已保存速度差异对比图")
    else:
        plt.show()
    
    plt.close()



def main():
    """主函数"""
    args = parse_args()
    
    # 加载两个CSV文件
    data1 = load_data(args.file1)
    data2 = load_data(args.file2)
    
    if data1 is None or data2 is None:
        print("数据加载失败，无法进行比较")
        return
    
    # 对齐数据
    aligned_data1, aligned_data2 = align_data(data1, data2, args.time_threshold)
    
    if len(aligned_data1) == 0:
        print("没有找到足够的时间对齐数据点")
        return
    
    # 获取文件名（用于图例）
    file1_name = os.path.basename(args.file1).split('.')[0]
    file2_name = os.path.basename(args.file2).split('.')[0]
    
    # 绘制对比图
    plot_vel_diff_comparison(aligned_data1, aligned_data2, file1_name, file2_name, args)
    
    print("比较完成！")


if __name__ == "__main__":
    main()