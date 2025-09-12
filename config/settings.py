# config/settings.py
from dataclasses import dataclass

@dataclass
class MediaPipeConfig:
    """MediaPipe姿态检测配置"""
    model_complexity: int = 1  # 模型复杂度（0=快，1=平衡，2=准）
    min_detection_confidence: float = 0.5  # 检测置信度阈值
    min_tracking_confidence: float = 0.5   # 跟踪置信度阈值

@dataclass
class VideoConfig:
    """视频输入/输出配置"""
    default_fps: float = 30.0  # 摄像头默认帧率
    output_video_codec: str = "mp4v"  # 输出视频编码（mp4格式）
    output_video_path: str = "output/processed_video.mp4"  # 输出视频路径
    output_csv_path: str = "output/joint_data.csv"  # 输出CSV路径

@dataclass
class AppConfig:
    """应用全局配置"""
    mediapipe: MediaPipeConfig = MediaPipeConfig()
    video: VideoConfig = VideoConfig()

# 实例化全局配置（供其他模块导入）
app_config = AppConfig()