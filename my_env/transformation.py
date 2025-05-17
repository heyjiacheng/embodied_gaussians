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

# Define the transformation matrix from OpenCV camera frame to Blender camera frame convention
# This matrix is used because the downstream scripts (like simple_body_builder.py)
# expect the input X_WC from cameras_tf.json to be in a "Blender camera" convention,
# such that their internal hardcoded transform (X_WC @ diag(1,-1,-1,1)) results in an OpenCV camera pose.
# So, if our current H is World->OpenCV_Cam, we need World->Blender_Cam.
# X_W_BlenderCam = X_W_OpenCVCam @ BlenderToOpenCV_Frame_Transfom_Inverse
# where BlenderToOpenCV_Frame_Transform is diag(1,-1,-1,1). This matrix is its own inverse.
OPENCV_CAM_TO_BLENDER_CAM_FRAME_TRANSFORM = np.array([
    [1,  0,  0,  0],
    [0,  1,  0,  0],
    [0,  0,  1,  0],
    [0,  0,  0,  1]
])

# ---------------- 1. 处理两个 D405 ---------------- #
# 直接把 easy_handeye/yaml 的四元数和平移抄进来
D405_CAM_PARAMS = {
    "130322272869": {   # D405-1 (可活动D405)
        "qw": 0.2710960382080267,
        "qx": -0.8359573827219756,
        "qy": 0.40253729802076466,
        "qz": -0.25621458983180295,
        "x": -0.6102768382685202,
        "y": -0.10164539839399017,
        "z": 0.4055639214811537,
    },
    "218622277783": {   # D405-2 (不可活动D405)
        "qw": 0.11393947646559957,
        "qx": -0.14774032459144876,
        "qy": 0.8895386138994323,
        "qz": -0.41702715328169576,
        "x": -0.2740167007047831,
        "y": 0.2801206234667821,
        "z": 0.36200378830666324,
    },
}

def d405_to_json_block():
    block = {}
    for serial, p in D405_CAM_PARAMS.items():
        H = homogeneous_from_quat(**p)
        # Assuming H is World->OpenCVCamera, convert to World->BlenderCamera for script input
        H = H @ OPENCV_CAM_TO_BLENDER_CAM_FRAME_TRANSFORM
        block[serial] = {"X_WT": format_matrix(H)}
    return block

# ---------------- 2. 处理腕端 D435 ---------------- #
# Tool→Camera 标定结果
T2C_QW, T2C_QX, T2C_QY, T2C_QZ = 0.9997083141872938, -0.023555480691573528, -0.0034626823830495868, 0.004054097298058542
T2C_t = np.array([-0.02114033416427041, -0.13959742644477738, 0.0334640815124871906])
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

    # Assuming H_B_C is Base->OpenCVCamera, convert to Base->BlenderCamera for script input
    H_B_C = H_B_C @ OPENCV_CAM_TO_BLENDER_CAM_FRAME_TRANSFORM

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
