from dataclasses import dataclass
from typing import Optional

@dataclass
class JointData:
    """结构化关节数据（包含所有主要关节的坐标、角度、角速度及元数据）"""
    # ------------------------------
    # 元数据（必传）
    # ------------------------------
    frame_idx: int          # 帧索引（来自FrameData）
    timestamp: float        # 帧时间戳（秒，来自FrameData）
    fps: float              # 输入源帧率（来自FrameData）
    
    # ------------------------------
    # 颈部（0号关键点：鼻子）
    # ------------------------------
    neck_x: Optional[float] = None  # 归一化x坐标（0-1）
    neck_y: Optional[float] = None  # 归一化y坐标（0-1）
    
    # ------------------------------
    # 肩部（11号：左，12号：右）
    # ------------------------------
    left_shoulder_x: Optional[float] = None
    left_shoulder_y: Optional[float] = None
    left_shoulder_angle: Optional[float] = None  # 颈→肩→肘的夹角（°）
    left_shoulder_velocity: Optional[float] = None  # 肩部角度角速度（°/s）
    
    right_shoulder_x: Optional[float] = None
    right_shoulder_y: Optional[float] = None
    right_shoulder_angle: Optional[float] = None  # 颈→肩→肘的夹角（°）
    right_shoulder_velocity: Optional[float] = None  # 肩部角度角速度（°/s）
    
    # ------------------------------
    # 肘部（13号：左，14号：右）
    # ------------------------------
    left_elbow_x: Optional[float] = None
    left_elbow_y: Optional[float] = None
    left_elbow_angle: Optional[float] = None  # 肩→肘→腕的夹角（°）
    left_elbow_velocity: Optional[float] = None  # 肘部角度角速度（°/s）
    
    right_elbow_x: Optional[float] = None
    right_elbow_y: Optional[float] = None
    right_elbow_angle: Optional[float] = None  # 肩→肘→腕的夹角（°）
    right_elbow_velocity: Optional[float] = None  # 肘部角度角速度（°/s）
    
    # ------------------------------
    # 腕部（15号：左，16号：右）
    # ------------------------------
    left_wrist_x: Optional[float] = None
    left_wrist_y: Optional[float] = None
    left_wrist_angle: Optional[float] = None  # 肘→腕→指的夹角（°）
    left_wrist_velocity: Optional[float] = None  # 腕部角度角速度（°/s）
    
    right_wrist_x: Optional[float] = None
    right_wrist_y: Optional[float] = None
    right_wrist_angle: Optional[float] = None  # 肘→腕→指的夹角（°）
    right_wrist_velocity: Optional[float] = None  # 腕部角度角速度（°/s）
    
    # ------------------------------
    # 手部（17号：左小指，18号：右拇指）- 用于腕部角度计算
    # ------------------------------
    left_thumb_x: Optional[float] = None  # 左小指坐标（辅助计算腕部角度）
    left_thumb_y: Optional[float] = None
    right_thumb_x: Optional[float] = None  # 右拇指坐标（辅助计算腕部角度）
    right_thumb_y: Optional[float] = None
    
    # ------------------------------
    # 髋部（23号：左，24号：右）
    # ------------------------------
    left_hip_x: Optional[float] = None
    left_hip_y: Optional[float] = None
    left_hip_angle: Optional[float] = None  # 髋→膝→踝的夹角（°）
    left_hip_velocity: Optional[float] = None  # 髋部角度角速度（°/s）
    
    right_hip_x: Optional[float] = None
    right_hip_y: Optional[float] = None
    right_hip_angle: Optional[float] = None  # 髋→膝→踝的夹角（°）
    right_hip_velocity: Optional[float] = None  # 髋部角度角速度（°/s）
    
    # ------------------------------
    # 膝部（25号：左，26号：右）
    # ------------------------------
    left_knee_x: Optional[float] = None
    left_knee_y: Optional[float] = None
    left_knee_angle: Optional[float] = None  # 膝→踝→脚跟的夹角（°）
    left_knee_velocity: Optional[float] = None  # 膝部角度角速度（°/s）
    
    right_knee_x: Optional[float] = None
    right_knee_y: Optional[float] = None
    right_knee_angle: Optional[float] = None  # 膝→踝→脚跟的夹角（°）
    right_knee_velocity: Optional[float] = None  # 膝部角度角速度（°/s）
    
    # ------------------------------
    # 踝部（27号：左，28号：右）
    # ------------------------------
    left_ankle_x: Optional[float] = None
    left_ankle_y: Optional[float] = None
    left_ankle_angle: Optional[float] = None  # 踝→脚跟→脚趾的夹角（°）
    left_ankle_velocity: Optional[float] = None  # 踝部角度角速度（°/s）
    
    right_ankle_x: Optional[float] = None
    right_ankle_y: Optional[float] = None
    right_ankle_angle: Optional[float] = None  # 踝→脚跟→脚趾的夹角（°）
    right_ankle_velocity: Optional[float] = None  # 踝部角度角速度（°/s）
    
    # ------------------------------
    # 脚跟（29号：左，30号：右）- 辅助计算膝部角度
    # ------------------------------
    left_heel_x: Optional[float] = None
    left_heel_y: Optional[float] = None
    right_heel_x: Optional[float] = None
    right_heel_y: Optional[float] = None
    
    # ------------------------------
    # 脚趾（31号：左，32号：右）- 辅助计算踝部角度
    # ------------------------------
    left_toe_x: Optional[float] = None
    left_toe_y: Optional[float] = None
    right_toe_x: Optional[float] = None
    right_toe_y: Optional[float] = None