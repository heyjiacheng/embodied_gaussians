#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成相机外参文件 cameras_tf.json：
{
    "234322306517": { "X_WT": [...] },   # D405-1  (eye-on-base)
    "220422302296": { "X_WT": [...] },   # D405-2  (eye-on-base)
    "234222302164": { "X_WT": [...] }    # D435    (eye-in-hand → Base 坐标)
}
"""

import math
import json
import numpy as np
from datetime import datetime

# ----------------  RTDE （腕端 D435 用） ---------------- #
try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:
    raise ImportError("请先 pip install ur_rtde")

ROBOT_IP = "192.168.1.60"           # ← 改成你的 UR5 IP
D435_SERIAL = "819612070593"        # ← 改成你的 D435 序列号/键名
OUTPUT_JSON = "cameras_tf.json"

# ------------- 公共数学工具函数 ------------- #
def quat_to_rotmat(qw, qx, qy, qz):
    """(qw,qx,qy,qz) → 3×3 旋转矩阵（ROS xyzw 顺序）"""
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)],
    ])

def axis_angle_to_rotmat(rx, ry, rz):
    """UR Pose 的 (Rx,Ry,Rz) → 3×3 旋转矩阵"""
    theta = math.sqrt(rx*rx + ry*ry + rz*rz)
    if theta < 1e-12:
        return np.eye(3)
    kx, ky, kz = rx/theta, ry/theta, rz/theta
    K = np.array([[0, -kz, ky],
                  [kz, 0, -kx],
                  [-ky, kx, 0]])
    return np.eye(3) + math.sin(theta) * K + (1-math.cos(theta)) * (K @ K)

def homogeneous_from_quat(qw, qx, qy, qz, x, y, z):
    H = np.eye(4)
    H[:3, :3] = quat_to_rotmat(qw, qx, qy, qz)
    H[:3,  3] = [x, y, z]
    return H

def format_matrix(mat, precision=12):
    """numpy 4×4 → Python list，保留 precision 位小数"""
    return [[round(float(v), precision) for v in row] for row in mat]

# ---------------- 1. 处理两个 D405 ---------------- #
# 直接把 easy_handeye/yaml 的四元数和平移抄进来
D405_CAM_PARAMS = {
    "130322272869": {   # D405-1
        "qw": 0.9260858261288483,
        "qx": -0.18802498858716796,
        "qy": 0.27798514763053617,
        "qz": 0.1724410160225025,
        "x": -0.5643474061541216,
        "y": -0.08784485885453652,
        "z": 0.38521996783967005,
    },
    "218622277783": {   # D405-2
        "qw": 0.8139158987083844,
        "qx": 0.16906151376184375,
        "qy": 0.2439698001163062,
        "qz": -0.49943753465822605,
        "x": -0.27250514982855756,
        "y": 0.28462689756862836,
        "z": 0.3575306266789925,
    },
}

def d405_to_json_block():
    block = {}
    for serial, p in D405_CAM_PARAMS.items():
        H = homogeneous_from_quat(**p)
        block[serial] = {"X_WT": format_matrix(H)}
    return block

# ---------------- 2. 处理腕端 D435 ---------------- #
# Tool→Camera 标定结果
T2C_QW, T2C_QX, T2C_QY, T2C_QZ = 0.9997931880670249, -0.01124135272524911, -0.006371957122653482, -0.015703860866297818
T2C_t = np.array([-0.03016181148276258, -0.1084391178977393, 0.021111178086654747])
R_T_C  = quat_to_rotmat(T2C_QW, T2C_QX, T2C_QY, T2C_QZ)

def get_robot_pose(ip):
    """实时读取 Base→Tool 的 6D Pose [x,y,z,Rx,Ry,Rz]"""
    rtde_control = RTDEControlInterface(ip)
    rtde_receive = RTDEReceiveInterface(ip)
    try:
        raw_pose = rtde_receive.getActualTCPPose()
        task_pose = raw_pose[:3] + [-v for v in raw_pose[3:]]   # Rx,Ry,Rz 取反
        return task_pose
    finally:
        rtde_control.stopScript()

def d435_to_json_block(ip):
    # ---- 读取 Base→Tool ----
    x_B_T, y_B_T, z_B_T, Rx, Ry, Rz = get_robot_pose(ip)
    R_B_T = axis_angle_to_rotmat(Rx, Ry, Rz)
    t_B_T = np.array([x_B_T, y_B_T, z_B_T])

    # ---- 拼接 Base→Camera ----
    R_B_C = R_B_T @ R_T_C
    t_B_C = t_B_T + R_B_T @ T2C_t
    H_B_C = np.eye(4)
    H_B_C[:3, :3] = R_B_C
    H_B_C[:3,  3] = t_B_C

    return {D435_SERIAL: {"X_WT": format_matrix(H_B_C)}}

# ---------------- 3. 主入口 ---------------- #
if __name__ == "__main__":
    result = {}
    result.update(d405_to_json_block())          # 两台 D405
    result.update(d435_to_json_block(ROBOT_IP))  # 一台 D435

    # 写入 json
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"[{datetime.now().isoformat(timespec='seconds')}] 已生成 {OUTPUT_JSON}")
