# recog/med_processor.py
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from config.settings import app_config
# 从video_io的data_classes导入FrameData（而不是input_handler！）
from video_io.data_classes import FrameData  # 关键！消除循环导入
from .data_classes import JointData  # 从本模块导入JointData

# 初始化MediaPipe工具（全局单例，避免重复创建）
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # 可选，用于调试绘制

class MediaPipeProcessor:
    def __init__(self):
        # MediaPipe模型配置
        self.model_complexity = app_config.mediapipe.model_complexity
        self.min_detection_confidence = app_config.mediapipe.min_detection_confidence
        self.min_tracking_confidence = app_config.mediapipe.min_tracking_confidence
        
        # 关节关键点索引映射（MediaPipe Pose 33个关键点）
        self.joint_index_map = {
            "left_shoulder": 11, "left_elbow": 13, "left_wrist": 15,
            "right_shoulder": 12, "right_elbow": 14, "right_wrist": 16,
            "left_hip": 23, "left_knee": 25, "left_ankle": 27,
            "right_hip": 24, "right_knee": 26, "right_ankle": 28
        }
        
        # 角度计算所需的关节三点索引（a: 近端, b: 关节中心, c: 远端）
        self.angle_joint_indices = {
            "left_elbow": (11, 13, 15),   # 左肩→左肘→左腕
            "right_elbow": (12, 14, 16),  # 右肩→右肘→右腕
            "left_knee": (23, 25, 27),    # 左髋→左膝→左踝
            "right_knee": (24, 26, 28)    # 右髋→右膝→右踝
        }
        
        # 可见性阈值（低于此值的关键点视为无效）
        self.visibility_threshold = 0.5
        # 缓存上一帧JointData（用于计算角速度）
        self.prev_joint_data: Optional[JointData] = None

    def process_frame(self, frame_data: FrameData) -> JointData:
        """处理一帧FrameData，返回结构化JointData（即使未检测到人体）"""
        # 1. 转换BGR→RGB（MediaPipe需要RGB输入）
        rgb_frame = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
        
        # 2. 运行MediaPipe Pose检测
        with mp_pose.Pose(
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        ) as pose:
            results = pose.process(rgb_frame)
        
        # 3. 初始化JointData（元数据必传，其他字段默认None）
        joint_data = JointData(
            frame_idx=frame_data.frame_idx,
            timestamp=frame_data.timestamp,
            fps=frame_data.fps
        )
        
        # 4. 检测到人体关键点：填充坐标、角度、角速度
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 提取关节坐标（带可见性过滤）
            joint_coords = self._extract_joint_coords(landmarks)
            for key, value in joint_coords.items():
                setattr(joint_data, key, value)  # 填充坐标字段（如left_shoulder_x）
            
            # 计算关节角度
            joint_angles = self._calculate_joint_angles(landmarks)
            for key, value in joint_angles.items():
                setattr(joint_data, f"{key}_angle", value)  # 填充角度字段（如left_elbow_angle）
            
            # 计算关节角速度
            joint_velocities = self._calculate_joint_velocities(joint_angles, frame_data.timestamp)
            for key, value in joint_velocities.items():
                setattr(joint_data, f"{key}_velocity", value)  # 填充角速度字段（如left_elbow_velocity）
            
            # 缓存当前关节数据（用于下一帧计算角速度）
            self.prev_joint_data = joint_data
        else:
            # 未检测到人体：重置缓存
            self.prev_joint_data = None
        
        # 5. 强制返回JointData（即使未检测到人体）
        return joint_data

    def _extract_joint_coords(self, landmarks: List[mp.solutions.pose.PoseLandmark]) -> Dict[str, Optional[float]]:
        """提取关节点坐标（带可见性过滤）"""
        coords = {}
        for joint_name, idx in self.joint_index_map.items():
            landmark = landmarks[idx]
            # 可见性低于阈值，坐标设为None
            if landmark.visibility < self.visibility_threshold:
                coords[f"{joint_name}_x"] = None
                coords[f"{joint_name}_y"] = None
            else:
                coords[f"{joint_name}_x"] = landmark.x
                coords[f"{joint_name}_y"] = landmark.y
        return coords

    def _calculate_joint_angles(self, landmarks: List[mp.solutions.pose.PoseLandmark]) -> Dict[str, Optional[float]]:
        """计算关节角度（度），基于三点向量夹角"""
        angles = {}
        for joint_name, (a_idx, b_idx, c_idx) in self.angle_joint_indices.items():
            a = landmarks[a_idx]
            b = landmarks[b_idx]
            c = landmarks[c_idx]
            
            # 任意一点可见性不足，角度设为None
            if (a.visibility < self.visibility_threshold or
                b.visibility < self.visibility_threshold or
                c.visibility < self.visibility_threshold):
                angles[joint_name] = None
                continue
            
            # 计算向量ab和cb（注意：b是关节中心，向量方向为【远端→中心】）
            vec_ab = np.array([a.x - b.x, a.y - b.y])  # 近端→中心
            vec_cb = np.array([c.x - b.x, c.y - b.y])  # 远端→中心
            
            # 向量长度为0（异常情况），角度设为None
            mag_ab = np.linalg.norm(vec_ab)
            mag_cb = np.linalg.norm(vec_cb)
            if mag_ab < 1e-6 or mag_cb < 1e-6:
                angles[joint_name] = None
                continue
            
            # 点积公式计算夹角（弧度转度）
            dot_product = np.dot(vec_ab, vec_cb)
            cos_theta = np.clip(dot_product / (mag_ab * mag_cb), -1.0, 1.0)  # 避免数值溢出
            theta_deg = np.degrees(np.arccos(cos_theta))
            angles[joint_name] = round(theta_deg, 2)  # 保留2位小数
        
        return angles

    def _calculate_joint_velocities(self, current_angles: Dict[str, Optional[float]], current_timestamp: float) -> Dict[str, Optional[float]]:
        """计算关节角速度（度/秒）"""
        velocities = {}
        # 无历史数据（第一帧或上一帧未检测到），角速度设为None
        if not self.prev_joint_data:
            for joint_name in current_angles.keys():
                velocities[joint_name] = None
            return velocities
        
        # 计算时间差（当前帧时间戳 - 上一帧时间戳）
        delta_time = current_timestamp - self.prev_joint_data.timestamp
        if delta_time <= 0:  # 时间差无效（帧率异常）
            for joint_name in current_angles.keys():
                velocities[joint_name] = None
            return velocities
        
        # 计算每个关节的角速度
        for joint_name, current_angle in current_angles.items():
            prev_angle = getattr(self.prev_joint_data, f"{joint_name}_angle", None)
            # 当前或上一帧角度无效，角速度设为None
            if current_angle is None or prev_angle is None:
                velocities[joint_name] = None
            else:
                # 角速度 = |当前角度 - 上一帧角度| / 时间差
                delta_angle = abs(current_angle - prev_angle)
                velocities[joint_name] = round(delta_angle / delta_time, 2)  # 保留2位小数
        
        return velocities

    def _extract_joint_coords(self, landmarks: List[mp.solutions.pose.PoseLandmark]) -> Dict[str, Optional[float]]:
        """提取关节点坐标（封装为独立方法，避免重复代码）"""
        coords = {}
        for joint_name, idx in self.joint_index_map.items():
            landmark = landmarks[idx]
            if landmark.visibility >= self.visibility_threshold:
                coords[f"{joint_name}_x"] = round(landmark.x, 4)  # 保留4位小数
                coords[f"{joint_name}_y"] = round(landmark.y, 4)
            else:
                coords[f"{joint_name}_x"] = None
                coords[f"{joint_name}_y"] = None
        return coords