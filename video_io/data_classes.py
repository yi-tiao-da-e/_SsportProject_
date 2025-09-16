# video_io/data_classes.py
from dataclasses import dataclass
import cv2

@dataclass
class FrameData:
    """结构化帧数据（独立存储，避免循环导入）"""
    frame: cv2.Mat  # BGR格式原始帧
    timestamp: float  # 帧捕获时间（秒）
    fps: float  # 输入源帧率
    frame_idx: int  # 帧索引（从0开始）