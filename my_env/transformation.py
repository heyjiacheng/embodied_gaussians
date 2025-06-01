"""
生成相机外参文件 cameras_tf.json：
{
    "130322272869": { "X_WT": [...] },   # D405-1  (可活动D405, eye-on-base)
    "218622277783": { "X_WT": [...] },   # D405-2  (不可活动D405, eye-on-base)
    "819612070593": { "X_WT": [...] }    # D435    (eye-in-hand → Base 坐标)
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
    [0,  -1,  0,  0],
    [0,  0,  -1,  0],
    [0,  0,  0,  1]
])

# ---------------- 1. 处理两个 D405 ---------------- #
# 直接把 easy_handeye/yaml 的四元数和平移抄进来
D405_CAM_PARAMS = {
    "130322272869": {   # D405-1 (可活动D405)
        "qw": 0.28614786206366677,
        "qx": -0.7643297850692737,
        "qy": 0.555472924317085,
        "qz": -0.15927715166644857,
        "x": -0.4415274123777343,
        "y": -0.1821370995983088,
        "z": 0.4017921406203354,
    },
    "218622277783": {   # D405-2 (不可活动D405)
        "qw": 0.06116262484577563,
        "qx": -0.39685481444253107,
        "qy": 0.860804158818786,
        "qz": -0.312700479270565,
        "x": -0.28901135627788893,
        "y": 0.3394417027837151,
        "z": 0.4351628755274338,
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
T2C_QW, T2C_QX, T2C_QY, T2C_QZ = 0.9983575560473741, -0.013281870667153105, -0.008029558586882923, -0.05514805874728528
T2C_t = np.array([-0.027174419406698548, -0.11432936025311663, 0.02828021481659389])
R_T_C  = quat_to_rotmat(T2C_QW, T2C_QX, T2C_QY, T2C_QZ)

def get_robot_pose(ip):
    """实时读取 Base→Tool 的 6D Pose [x,y,z,Rx,Ry,Rz]"""
    rtde_control = RTDEControlInterface(ip)
    rtde_receive = RTDEReceiveInterface(ip)
    try:
        raw_pose = rtde_receive.getActualTCPPose()
        # 不对旋转角度取负号，直接使用原始值
        task_pose = raw_pose  # [x, y, z, Rx, Ry, Rz]
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
