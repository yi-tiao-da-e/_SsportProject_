# recog/data_classes.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class JointData:
    """结构化关节数据（包含坐标、角度、角速度及元数据）"""
    # 元数据
    frame_idx: int          # 帧索引
    timestamp: float        # 帧时间戳（秒）
    fps: float              # 输入源帧率
    # 关节角度（度）
    left_elbow_angle: Optional[float] = None
    right_elbow_angle: Optional[float] = None
    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None
    # 关节角速度（度/秒）
    left_elbow_velocity: Optional[float] = None
    right_elbow_velocity: Optional[float] = None
    left_knee_velocity: Optional[float] = None
    right_knee_velocity: Optional[float] = None
    # 关节点坐标（归一化，0-1）
    left_shoulder_x: Optional[float] = None
    left_shoulder_y: Optional[float] = None
    left_elbow_x: Optional[float] = None
    left_elbow_y: Optional[float] = None
    left_wrist_x: Optional[float] = None
    left_wrist_y: Optional[float] = None
    right_shoulder_x: Optional[float] = None
    right_shoulder_y: Optional[float] = None
    right_elbow_x: Optional[float] = None
    right_elbow_y: Optional[float] = None
    right_wrist_x: Optional[float] = None
    right_wrist_y: Optional[float] = None
    left_hip_x: Optional[float] = None
    left_hip_y: Optional[float] = None
    left_knee_x: Optional[float] = None
    left_knee_y: Optional[float] = None
    left_ankle_x: Optional[float] = None
    left_ankle_y: Optional[float] = None
    right_hip_x: Optional[float] = None
    right_hip_y: Optional[float] = None
    right_knee_x: Optional[float] = None
    right_knee_y: Optional[float] = None
    right_ankle_x: Optional[float] = None
    right_ankle_y: Optional[float] = None