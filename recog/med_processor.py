import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from config.settings import app_config  # 导入全局配置
from video_io.input_handler import FrameData  # 导入输入数据结构

# 初始化MediaPipe Pose模型（类级别的初始化，避免重复创建）
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # 可选，用于调试

@dataclass
class JointData:
    # 【省略元数据、角度、角速度字段，同上述定义】
    pass

class MediaPipeProcessor:
    def __init__(self):
        # MediaPipe Pose模型配置（来自settings）
        self.pose = mp_pose.Pose(
            model_complexity=app_config.mediapipe.model_complexity,
            min_detection_confidence=app_config.mediapipe.min_detection_confidence,
            min_tracking_confidence=app_config.mediapipe.min_tracking_confidence
        )
        # 【关键】关节点索引映射（MediaPipe Pose 33个关键点）
        # 格式：(关节名称, 索引)，用于快速提取
        self.joint_index_map = {
            # 左上肢
            "left_shoulder": 11,
            "left_elbow": 13,
            "left_wrist": 15,
            # 右上肢
            "right_shoulder": 12,
            "right_elbow": 14,
            "right_wrist": 16,
            # 左下肢
            "left_hip": 23,
            "left_knee": 25,
            "left_ankle": 27,
            # 右下肢
            "right_hip": 24,
            "right_knee": 26,
            "right_ankle": 28
        }
        # 可见性阈值（低于此值的关键点视为无效）
        self.visibility_threshold = 0.5
        # 缓存上一帧JointData（用于计算角速度）
        self.prev_joint_data: Optional[JointData] = None

    def process_frame(self, frame_data: FrameData) -> Optional[JointData]:
        """
        处理一帧FrameData，提取关节点坐标、计算角度/角速度，返回结构化JointData
        :param frame_data: 输入帧数据（包含帧图像、时间戳、帧率）
        :return: JointData（含坐标、角度、角速度），未检测到人体则返回None
        """
        # 1. 转换帧格式（BGR→RGB）
        rgb_frame = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
        
        # 2. 运行MediaPipe Pose检测
        with mp_pose.Pose(
            model_complexity=app_config.mediapipe.model_complexity,
            min_detection_confidence=app_config.mediapipe.min_detection_confidence,
            min_tracking_confidence=app_config.mediapipe.min_tracking_confidence
        ) as pose:
            results = pose.process(rgb_frame)
        
        # 3. 检查是否检测到人体
        if not results.pose_landmarks:
            return None  # 未检测到人体，返回None
        
        # 4. 提取landmarks列表（可索引）
        landmarks = results.pose_landmarks.landmark  # type: List[mp.solutions.pose.PoseLandmark]
        
        # 5. 提取关节点坐标（归一化，0-1）
        joint_coords = {}
        for joint_name, idx in self.joint_index_map.items():
            # 获取关键点的x/y坐标与visibility
            landmark = landmarks[idx]
            x = landmark.x
            y = landmark.y
            visibility = landmark.visibility
            
            # 可见性过滤：低于阈值则设为None
            if visibility < self.visibility_threshold:
                joint_coords[f"{joint_name}_x"] = None
                joint_coords[f"{joint_name}_y"] = None
            else:
                joint_coords[f"{joint_name}_x"] = x
                joint_coords[f"{joint_name}_y"] = y
        
        # 6. 计算关节角度（基于坐标，使用向量点积）
        joint_angles = self._calculate_joint_angles(landmarks)
        
        # 7. 计算关节角速度（基于上一帧角度）
        joint_velocities = self._calculate_joint_velocities(joint_angles)
        
        # 8. 构造JointData对象（整合坐标、角度、角速度）
        joint_data = JointData(
            # 元数据（来自FrameData）
            frame_idx=frame_data.frame_idx,
            timestamp=frame_data.timestamp,
            fps=frame_data.fps,
            # 关节点坐标（来自joint_coords）
            left_shoulder_x=joint_coords["left_shoulder_x"],
            left_shoulder_y=joint_coords["left_shoulder_y"],
            left_elbow_x=joint_coords["left_elbow_x"],
            left_elbow_y=joint_coords["left_elbow_y"],
            left_wrist_x=joint_coords["left_wrist_x"],
            left_wrist_y=joint_coords["left_wrist_y"],
            right_shoulder_x=joint_coords["right_shoulder_x"],
            right_shoulder_y=joint_coords["right_shoulder_y"],
            right_elbow_x=joint_coords["right_elbow_x"],
            right_elbow_y=joint_coords["right_elbow_y"],
            right_wrist_x=joint_coords["right_wrist_x"],
            right_wrist_y=joint_coords["right_wrist_y"],
            left_hip_x=joint_coords["left_hip_x"],
            left_hip_y=joint_coords["left_hip_y"],
            left_knee_x=joint_coords["left_knee_x"],
            left_knee_y=joint_coords["left_knee_y"],
            left_ankle_x=joint_coords["left_ankle_x"],
            left_ankle_y=joint_coords["left_ankle_y"],
            right_hip_x=joint_coords["right_hip_x"],
            right_hip_y=joint_coords["right_hip_y"],
            right_knee_x=joint_coords["right_knee_x"],
            right_knee_y=joint_coords["right_knee_y"],
            right_ankle_x=joint_coords["right_ankle_x"],
            right_ankle_y=joint_coords["right_ankle_y"],
            # 关节角度（来自joint_angles）
            left_elbow_angle=joint_angles["left_elbow"],
            right_elbow_angle=joint_angles["right_elbow"],
            left_knee_angle=joint_angles["left_knee"],
            right_knee_angle=joint_angles["right_knee"],
            # 关节角速度（来自joint_velocities）
            left_elbow_velocity=joint_velocities["left_elbow"],
            right_elbow_velocity=joint_velocities["right_elbow"],
            left_knee_velocity=joint_velocities["left_knee"],
            right_knee_velocity=joint_velocities["right_knee"]
        )
        
        # 9. 缓存当前JointData（用于下一帧计算角速度）
        self.prev_joint_data = joint_data
        
        return joint_data

    def _calculate_joint_angles(self, landmarks: List[mp.solutions.pose.PoseLandmark]) -> dict:
        """
        计算膝/肘关节角度（度），基于三个关键点的向量夹角
        :param landmarks: MediaPipe Pose关键点列表
        :return: 角度字典（如{"left_elbow": 120.5}）
        """
        # 定义关节的三个关键点索引（中心关节为中间索引）
        joint_angle_indices = {
            "left_elbow": (11, 13, 15),   # 左肩(11) → 左肘(13) → 左腕(15)
            "right_elbow": (12, 14, 16),  # 右肩(12) → 右肘(14) → 右腕(16)
            "left_knee": (23, 25, 27),    # 左髋(23) → 左膝(25) → 左踝(27)
            "right_knee": (24, 26, 28)    # 右髋(24) → 右膝(26) → 右踝(28)
        }
        
        angles = {}
        for joint_name, (a_idx, b_idx, c_idx) in joint_angle_indices.items():
            # 提取三个关键点的坐标（已过滤可见性）
            a = landmarks[a_idx]
            b = landmarks[b_idx]
            c = landmarks[c_idx]
            
            # 检查可见性（避免无效计算）
            if a.visibility < self.visibility_threshold or b.visibility < self.visibility_threshold or c.visibility < self.visibility_threshold:
                angles[joint_name] = None
                continue
            
            # 转换为向量（使用归一化坐标）
            vec_ab = np.array([a.x - b.x, a.y - b.y])  # 肩→肘向量
            vec_cb = np.array([c.x - b.x, c.y - b.y])  # 腕→肘向量（或踝→膝）
            
            # 计算向量夹角（点积公式）
            dot_product = np.dot(vec_ab, vec_cb)
            mag_ab = np.linalg.norm(vec_ab)  # 向量长度
            mag_cb = np.linalg.norm(vec_cb)
            
            if mag_ab == 0 or mag_cb == 0:
                angles[joint_name] = None  # 避免除以零
                continue
            
            cos_theta = dot_product / (mag_ab * mag_cb)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 限制范围（避免数值误差）
            theta_rad = np.arccos(cos_theta)  # 弧度
            theta_deg = np.degrees(theta_rad)  # 转换为度
            
            angles[joint_name] = theta_deg
        
        return angles

    def _calculate_joint_velocities(self, current_angles: dict) -> dict:
        """
        计算关节角速度（度/秒），对比上一帧角度
        :param current_angles: 当前帧关节角度字典
        :return: 角速度字典（如{"left_elbow": 30.0}）
        """
        velocities = {}
        if not self.prev_joint_data:
            # 第一帧，无历史数据，角速度设为None
            for joint_name in current_angles.keys():
                velocities[joint_name] = None
            return velocities
        
        # 计算时间差（当前帧时间戳 - 上一帧时间戳）
        delta_time = self.prev_joint_data.timestamp - self.prev_joint_data.timestamp
        if delta_time <= 0:
            # 时间差无效（如帧率异常），角速度设为None
            for joint_name in current_angles.keys():
                velocities[joint_name] = None
            return velocities
        
        # 遍历每个关节，计算角速度
        for joint_name in current_angles.keys():
            current_angle = current_angles[joint_name]
            prev_angle = getattr(self.prev_joint_data, f"{joint_name}_angle", None)
            
            if current_angle is None or prev_angle is None:
                velocities[joint_name] = None
                continue
            
            # 角速度 = 角度变化量 / 时间差（取绝对值）
            delta_angle = abs(current_angle - prev_angle)
            velocity = delta_angle / delta_time
            
            velocities[joint_name] = velocity
        
        return velocities

# 定义关节数据结构（供presentation使用）
@dataclass
class JointData:
    """结构化关节数据（包含角度、角速度及元数据）"""
    frame_idx: int          # 帧索引
    timestamp: float        # 帧时间戳（秒）
    fps: float              # 输入源帧率
    # 肘关节角度（度，None表示未检测到）
    left_elbow_angle: Optional[float]
    right_elbow_angle: Optional[float]
    # 膝关节角度（度，None表示未检测到）
    left_knee_angle: Optional[float]
    right_knee_angle: Optional[float]
    # 肘关节角速度（度/秒，None表示无法计算）
    left_elbow_velocity: Optional[float]
    right_elbow_velocity: Optional[float]
    # 膝关节角速度（度/秒，None表示无法计算）
    left_knee_velocity: Optional[float]
    right_knee_velocity: Optional[float]

class MediaPipeProcessor:
    def __init__(self):
        # 初始化MediaPipe Pose模型（使用settings中的配置）
        self.pose = mp.solutions.pose.Pose(
            model_complexity=app_config.mediapipe.model_complexity,
            min_detection_confidence=app_config.mediapipe.min_detection_confidence,
            min_tracking_confidence=app_config.mediapipe.min_tracking_confidence
        )
        # 关节关键点索引（MediaPipe Pose 33个关键点）
        # 格式：(a, b, c)，其中b是关节中心（如肘、膝）
        self.joint_indices = {
            "left_elbow": (11, 13, 15),   # 左肩(11) → 左肘(13) → 左腕(15)
            "right_elbow": (12, 14, 16),  # 右肩(12) → 右肘(14) → 右腕(16)
            "left_knee": (23, 25, 27),    # 左髋(23) → 左膝(25) → 左踝(27)
            "right_knee": (24, 26, 28)    # 右髋(24) → 右膝(26) → 右踝(28)
        }
        # 可见性阈值（低于该值的关键点视为无效）
        self.visibility_threshold = 0.5
        # 缓存上一帧的JointData（用于计算角速度）
        self.prev_joint_data: Optional[JointData] = None

    def _calculate_angle(self, landmarks: List[mp.solutions.pose.PoseLandmark], indices: tuple) -> Optional[float]:
        """
        计算关节角度（度），基于三个关键点的向量夹角
        :param landmarks: MediaPipe Pose landmarks列表
        :param indices: 三个关键点的索引（a, b, c）
        :return: 角度（度，0-180），若关键点可见性低则返回None
        """
        a_idx, b_idx, c_idx = indices
        # 提取关键点的坐标与可见性
        a = landmarks[a_idx]
        b = landmarks[b_idx]
        c = landmarks[c_idx]
        # 检查可见性
        if a.visibility < self.visibility_threshold or b.visibility < self.visibility_threshold or c.visibility < self.visibility_threshold:
            return None
        # 转换为向量（使用归一化坐标，不影响角度计算）
        vec_ab = np.array([a.x - b.x, a.y - b.y])  # 肩→肘向量
        vec_cb = np.array([c.x - b.x, c.y - b.y])  # 腕→肘向量（或踝→膝）
        # 计算向量夹角（点积公式）
        dot_product = np.dot(vec_ab, vec_cb)
        mag_ab = np.linalg.norm(vec_ab)  # 向量长度
        mag_cb = np.linalg.norm(vec_cb)
        if mag_ab == 0 or mag_cb == 0:
            return None  # 避免除以零
        cos_theta = dot_product / (mag_ab * mag_cb)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 限制范围（避免数值误差）
        theta_rad = np.arccos(cos_theta)  # 弧度
        theta_deg = np.degrees(theta_rad)  # 转换为度
        return theta_deg


@dataclass
class JointData:
    # 元数据
    frame_idx: int          # 帧索引
    timestamp: float        # 帧时间戳（秒）
    fps: float              # 输入源帧率
    # 关节角度（度）
    left_elbow_angle: Optional[float]
    right_elbow_angle: Optional[float]
    left_knee_angle: Optional[float]
    right_knee_angle: Optional[float]
    # 关节角速度（度/秒）
    left_elbow_velocity: Optional[float]
    right_elbow_velocity: Optional[float]
    left_knee_velocity: Optional[float]
    right_knee_velocity: Optional[float]
    # 关节点坐标（归一化，0-1，需从MediaPipe landmarks中提取）
    left_shoulder_x: Optional[float]
    left_shoulder_y: Optional[float]
    left_elbow_x: Optional[float]
    left_elbow_y: Optional[float]
    left_wrist_x: Optional[float]
    left_wrist_y: Optional[float]
    right_shoulder_x: Optional[float]
    right_shoulder_y: Optional[float]
    right_elbow_x: Optional[float]
    right_elbow_y: Optional[float]
    right_wrist_x: Optional[float]
    right_wrist_y: Optional[float]
    left_hip_x: Optional[float]
    left_hip_y: Optional[float]
    left_knee_x: Optional[float]
    left_knee_y: Optional[float]
    left_ankle_x: Optional[float]
    left_ankle_y: Optional[float]
    right_hip_x: Optional[float]
    right_hip_y: Optional[float]
    right_knee_x: Optional[float]
    right_knee_y: Optional[float]
    right_ankle_x: Optional[float]
    right_ankle_y: Optional[float]