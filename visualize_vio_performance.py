#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIO性能可视化脚本
用于可视化sqrt_vins生成的vio_performance.csv文件中的数据
支持位置、姿态、速度等多种数据的可视化
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VIO性能可视化脚本')
    parser.add_argument('--csv_file', type=str, default='/root/catkin_ws/src/open_vins/logs/vio_performance.csv',
                        help='CSV文件路径')
    parser.add_argument('--save', action='store_true', help='保存图表而不显示')
    parser.add_argument('--output_dir', type=str, default='/root/catkin_ws/src/open_vins/logs',
                        help='图表保存目录')
    parser.add_argument('--animate', action='store_true', help='启用3D轨迹动画')
    return parser.parse_args()


def load_data(csv_file):
    """加载CSV数据"""
    if not os.path.exists(csv_file):
        print(f"错误：文件 {csv_file} 不存在")
        exit(1)
    
    try:
        data = pd.read_csv(csv_file)
        print(f"成功加载数据：{len(data)} 行，{len(data.columns)} 列")
        return data
    except Exception as e:
        print(f"加载数据时出错：{e}")
        exit(1)


def plot_position_3d(data, args):
    """绘制3D位置轨迹"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹
    ax.plot(data['pos_x'], data['pos_y'], data['pos_z'], '-', linewidth=2, label='Position Trajectory')
    
    # 绘制起点和终点
    ax.scatter(data['pos_x'].iloc[0], data['pos_y'].iloc[0], data['pos_z'].iloc[0], 
               c='g', s=100, marker='o', label='Start')
    ax.scatter(data['pos_x'].iloc[-1], data['pos_y'].iloc[-1], data['pos_z'].iloc[-1], 
               c='r', s=100, marker='x', label='End')
    
    ax.set_title('3D Position Trajectory', fontsize=16)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_zlabel('Z Position (m)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    # 设置相同的坐标轴范围，使轨迹更清晰
    max_range = np.array([data['pos_x'].max()-data['pos_x'].min(), 
                         data['pos_y'].max()-data['pos_y'].min(), 
                         data['pos_z'].max()-data['pos_z'].min()]).max() / 2.0
    
    mid_x = (data['pos_x'].max()+data['pos_x'].min()) * 0.5
    mid_y = (data['pos_y'].max()+data['pos_y'].min()) * 0.5
    mid_z = (data['pos_z'].max()+data['pos_z'].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if args.save:
        plt.savefig(os.path.join(args.output_dir, 'position_3d.png'), dpi=300, bbox_inches='tight')
        print("已保存3D位置轨迹图")
    else:
        plt.show()
    
    plt.close()


def plot_position_2d(data, args):
    """绘制2D位置轨迹（X-Y, X-Z, Y-Z）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # X-Y平面
    axes[0].plot(data['pos_x'], data['pos_y'], '-', linewidth=2, color='blue')
    axes[0].scatter(data['pos_x'].iloc[0], data['pos_y'].iloc[0], c='g', s=50, marker='o', label='Start')
    axes[0].scatter(data['pos_x'].iloc[-1], data['pos_y'].iloc[-1], c='r', s=50, marker='x', label='End')
    axes[0].set_title('X-Y Position', fontsize=14)
    axes[0].set_xlabel('X (m)', fontsize=12)
    axes[0].set_ylabel('Y (m)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True)
    
    # X-Z平面
    axes[1].plot(data['pos_x'], data['pos_z'], '-', linewidth=2, color='green')
    axes[1].scatter(data['pos_x'].iloc[0], data['pos_z'].iloc[0], c='g', s=50, marker='o')
    axes[1].scatter(data['pos_x'].iloc[-1], data['pos_z'].iloc[-1], c='r', s=50, marker='x')
    axes[1].set_title('X-Z Position', fontsize=14)
    axes[1].set_xlabel('X (m)', fontsize=12)
    axes[1].set_ylabel('Z (m)', fontsize=12)
    axes[1].grid(True)
    
    # Y-Z平面
    axes[2].plot(data['pos_y'], data['pos_z'], '-', linewidth=2, color='red')
    axes[2].scatter(data['pos_y'].iloc[0], data['pos_z'].iloc[0], c='g', s=50, marker='o')
    axes[2].scatter(data['pos_y'].iloc[-1], data['pos_z'].iloc[-1], c='r', s=50, marker='x')
    axes[2].set_title('Y-Z Position', fontsize=14)
    axes[2].set_xlabel('Y (m)', fontsize=12)
    axes[2].set_ylabel('Z (m)', fontsize=12)
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if args.save:
        plt.savefig(os.path.join(args.output_dir, 'position_2d.png'), dpi=300, bbox_inches='tight')
        print("已保存2D位置轨迹图")
    else:
        plt.show()
    
    plt.close()


def plot_velocity_comparison(data, args):
    """绘制速度对比图（原始速度 vs 计算速度）"""
    # 计算时间序列
    time = data['timestamp'] - data['timestamp'].iloc[0]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # X方向速度
    axes[0].plot(time, data['raw_vel_x'], '-', linewidth=2, color='blue', label='Raw Velocity (X)')
    axes[0].plot(time, data['diff_vel_x'], '-', linewidth=2, color='red', label='Calculated Velocity (X)')
    axes[0].set_title('Velocity Comparison - X Direction', fontsize=14)
    axes[0].set_ylabel('Velocity (m/s)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True)
    
    # Y方向速度
    axes[1].plot(time, data['raw_vel_y'], '-', linewidth=2, color='blue', label='Raw Velocity (Y)')
    axes[1].plot(time, data['diff_vel_y'], '-', linewidth=2, color='red', label='Calculated Velocity (Y)')
    axes[1].set_title('Velocity Comparison - Y Direction', fontsize=14)
    axes[1].set_ylabel('Velocity (m/s)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True)
    
    # Z方向速度
    axes[2].plot(time, data['raw_vel_z'], '-', linewidth=2, color='blue', label='Raw Velocity (Z)')
    axes[2].plot(time, data['diff_vel_z'], '-', linewidth=2, color='red', label='Calculated Velocity (Z)')
    axes[2].set_title('Velocity Comparison - Z Direction', fontsize=14)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Velocity (m/s)', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if args.save:
        plt.savefig(os.path.join(args.output_dir, 'velocity_comparison.png'), dpi=300, bbox_inches='tight')
        print("已保存速度对比图")
    else:
        plt.show()
    
    plt.close()


def plot_orientation(data, args):
    """绘制姿态变化（四元数分量）"""
    # 计算时间序列
    time = data['timestamp'] - data['timestamp'].iloc[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    
    axes[0, 0].plot(time, data['ori_w'], '-', linewidth=2, color='blue')
    axes[0, 0].set_title('Quaternion W Component', fontsize=14)
    axes[0, 0].set_ylabel('W', fontsize=12)
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(time, data['ori_x'], '-', linewidth=2, color='green')
    axes[0, 1].set_title('Quaternion X Component', fontsize=14)
    axes[0, 1].set_ylabel('X', fontsize=12)
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(time, data['ori_y'], '-', linewidth=2, color='red')
    axes[1, 0].set_title('Quaternion Y Component', fontsize=14)
    axes[1, 0].set_ylabel('Y', fontsize=12)
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(time, data['ori_z'], '-', linewidth=2, color='purple')
    axes[1, 1].set_title('Quaternion Z Component', fontsize=14)
    axes[1, 1].set_xlabel('Time (s)', fontsize=12)
    axes[1, 1].set_ylabel('Z', fontsize=12)
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if args.save:
        plt.savefig(os.path.join(args.output_dir, 'orientation_quaternion.png'), dpi=300, bbox_inches='tight')
        print("已保存姿态变化图")
    else:
        plt.show()
    
    plt.close()


def plot_differences(data, args):
    """绘制速度差异大小的表格（X, Y, Z）"""
    # 计算时间序列
    time = data['timestamp'] - data['timestamp'].iloc[0]
    
    # 只创建一个图表，显示速度差异
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # 速度差异 X
    axes[0].plot(time, data['vel_diff_x'], '-', linewidth=2, color='blue', label='Velocity Diff (X)')
    axes[0].set_title('Velocity Difference - X Direction', fontsize=14)
    axes[0].set_ylabel('Difference (m/s)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True)
    
    # 速度差异 Y
    axes[1].plot(time, data['vel_diff_y'], '-', linewidth=2, color='green', label='Velocity Diff (Y)')
    axes[1].set_title('Velocity Difference - Y Direction', fontsize=14)
    axes[1].set_ylabel('Difference (m/s)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True)
    
    # 速度差异 Z
    axes[2].plot(time, data['vel_diff_z'], '-', linewidth=2, color='red', label='Velocity Diff (Z)')
    axes[2].set_title('Velocity Difference - Z Direction', fontsize=14)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Difference (m/s)', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if args.save:
        plt.savefig(os.path.join(args.output_dir, 'velocity_differences.png'), dpi=300, bbox_inches='tight')
        print("已保存速度差异图")
    else:
        plt.show()
    
    plt.close()


def animate_3d_trajectory(data, args):
    """创建3D轨迹动画"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴范围
    max_range = np.array([data['pos_x'].max()-data['pos_x'].min(), 
                         data['pos_y'].max()-data['pos_y'].min(), 
                         data['pos_z'].max()-data['pos_z'].min()]).max() / 2.0
    
    mid_x = (data['pos_x'].max()+data['pos_x'].min()) * 0.5
    mid_y = (data['pos_y'].max()+data['pos_y'].min()) * 0.5
    mid_z = (data['pos_z'].max()+data['pos_z'].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_title('3D Position Trajectory Animation', fontsize=16)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_zlabel('Z Position (m)', fontsize=12)
    ax.grid(True)
    
    # 初始化轨迹线和当前点
    line, = ax.plot([], [], [], '-', linewidth=2, color='blue')
    point, = ax.plot([], [], [], 'o', markersize=8, color='red')
    
    # 设置点的初始位置
    point.set_data([data['pos_x'].iloc[0]], [data['pos_y'].iloc[0]])
    point.set_3d_properties([data['pos_z'].iloc[0]])
    
    def update(frame):
        """动画更新函数"""
        if frame >= len(data):
            return line, point
        
        # 更新轨迹线
        line.set_data(data['pos_x'].iloc[:frame+1], data['pos_y'].iloc[:frame+1])
        line.set_3d_properties(data['pos_z'].iloc[:frame+1])
        
        # 更新当前点
        point.set_data([data['pos_x'].iloc[frame]], [data['pos_y'].iloc[frame]])
        point.set_3d_properties([data['pos_z'].iloc[frame]])
        
        return line, point
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(data), interval=50, blit=True)
    
    if args.save:
        # 保存动画为GIF
        ani.save(os.path.join(args.output_dir, 'trajectory_animation.gif'), writer='imagemagick', fps=20)
        print("已保存3D轨迹动画")
    else:
        plt.show()
    
    plt.close()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    if args.save and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 加载数据
    data = load_data(args.csv_file)
    
    # 绘制各种图表
    print("绘制3D位置轨迹...")
    plot_position_3d(data, args)
    
    print("绘制2D位置轨迹...")
    plot_position_2d(data, args)
    
    print("绘制速度对比图...")
    plot_velocity_comparison(data, args)
    
    print("绘制姿态变化图...")
    plot_orientation(data, args)
    
    print("绘制差异分析图...")
    plot_differences(data, args)
    
    # 如果启用动画
    if args.animate:
        print("创建3D轨迹动画...")
        animate_3d_trajectory(data, args)
    
    print("可视化完成！")


if __name__ == "__main__":
    main()