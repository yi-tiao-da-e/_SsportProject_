import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from config.settings import app_config
from video_io.data_classes import FrameData  # 从video_io导入FrameData
from .data_classes import JointData  # 从本模块导入更新后的JointData

# 初始化MediaPipe工具（全局单例，避免重复创建）
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # 可选，用于调试绘制

class MediaPipeProcessor:
    def __init__(self):
        # MediaPipe模型配置（从配置文件读取）
        self.model_complexity = app_config.mediapipe.model_complexity
        self.min_detection_confidence = app_config.mediapipe.min_detection_confidence
        self.min_tracking_confidence = app_config.mediapipe.min_tracking_confidence
        
        # ------------------------------
        # 扩展：全关节索引映射（MediaPipe Pose 33个关键点）
        # ------------------------------
        self.joint_index_map = {
            # 核心关节（必选）
            "neck": 0,                  # 0号：鼻子（颈部替代点）
            "left_shoulder": 11,        # 11号：左肩膀
            "right_shoulder": 12,       # 12号：右肩膀
            "left_elbow": 13,           # 13号：左肘
            "right_elbow": 14,          # 14号：右肘
            "left_wrist": 15,           # 15号：左腕
            "right_wrist": 16,          # 16号：右腕
            "left_hip": 23,             # 23号：左髋
            "right_hip": 24,            # 24号：右髋
            "left_knee": 25,            # 25号：左膝
            "right_knee": 26,           # 26号：右膝
            "left_ankle": 27,           # 27号：左踝
            "right_ankle": 28,          # 28号：右踝
            # 辅助关节（用于角度计算）
            "left_thumb": 17,           # 17号：左拇指
            "right_thumb": 18,          # 18号：右拇指
            "left_heel": 29,            # 29号：左脚跟
            "right_heel": 30,           # 30号：右脚跟
            "left_toe": 31,             # 31号：左脚趾
            "right_toe": 32,            # 32号：右脚趾
        }
        
        # ------------------------------
        # 扩展：全关节角度计算三点索引（近端→关节中心→远端）
        # ------------------------------
        self.angle_joint_indices = {
            # 肩部（脖子→肩→肘）
            "left_shoulder": (0, 11, 13),   # 0: 脖子（鼻子）→ 11: 左肩膀→ 13: 左肘
            "right_shoulder": (0, 12, 14),  # 0: 脖子→ 12: 右肩膀→ 14: 右肘
            # 肘部（肩→肘→腕）
            "left_elbow": (11, 13, 15),     # 11: 左肩膀→ 13: 左肘→ 15: 左腕
            "right_elbow": (12, 14, 16),    # 12: 右肩膀→ 14: 右肘→ 16: 右腕
            # 腕部（肘→腕→拇指）
            "left_wrist": (13, 15, 17),     # 13: 左肘→ 15: 左腕→ 17: 左拇指
            "right_wrist": (14, 16, 18),    # 14: 右肘→ 16: 右腕→ 18: 右拇指
            # 髋部（腰→髋→膝，腰=左右髋中点）
            "left_hip": (23, 24, 25),       
            "right_hip": (23, 24, 26),      
            # 膝部（髋→膝→踝）
            "left_knee": (23, 25, 27),      # 23: 左髋→ 25: 左膝→ 27: 左踝
            "right_knee": (24, 26, 28),     # 24: 右髋→ 26: 右膝→ 28: 右踝
            # 踝部（膝→踝→脚跟）
            "left_ankle": (25, 27, 29),     # 25: 左膝→ 27: 左踝→ 29: 左脚跟
            "right_ankle": (26, 28, 30),    # 26: 右膝→ 28: 右踝→ 30: 右脚跟
            # 脚趾（踝→脚跟→脚趾）
            "left_toe": (27, 29, 31),       # 27: 左踝→ 29: 左脚跟→ 31: 左脚趾
            "right_toe": (28, 30, 32),      # 28: 右踝→ 30: 右脚跟→ 32: 右脚趾
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
                setattr(joint_data, key, value)  # 填充坐标字段（如neck_x、left_shoulder_x）
            
            # 计算关节角度（带可见性过滤）
            joint_angles = self._calculate_joint_angles(landmarks)
            for key, value in joint_angles.items():
                setattr(joint_data, f"{key}_angle", value)  # 填充角度字段（如left_shoulder_angle）
            
            # 计算关节角速度（基于当前帧与上一帧的角度差）
            joint_velocities = self._calculate_joint_velocities(joint_angles, frame_data.timestamp)
            for key, value in joint_velocities.items():
                setattr(joint_data, f"{key}_velocity", value)  # 填充角速度字段（如left_shoulder_velocity）
            
            # 缓存当前关节数据（用于下一帧计算角速度）
            self.prev_joint_data = joint_data
        else:
            # 未检测到人体：重置缓存
            self.prev_joint_data = None
        
        # 5. 强制返回JointData（即使未检测到人体）
        return joint_data

    def _extract_joint_coords(self, landmarks: List[mp.solutions.pose.PoseLandmark]) -> Dict[str, Optional[float]]:
        """提取所有关节点坐标（带可见性过滤）"""
        coords = {}
        for joint_name, idx in self.joint_index_map.items():
            landmark = landmarks[idx]
            # 可见性低于阈值：坐标设为None
            if landmark.visibility < self.visibility_threshold:
                coords[f"{joint_name}_x"] = None
                coords[f"{joint_name}_y"] = None
            else:
                # 保留4位小数（减少数据量）
                coords[f"{joint_name}_x"] = round(landmark.x, 4)
                coords[f"{joint_name}_y"] = round(landmark.y, 4)
        return coords

    def _calculate_joint_angles(self, landmarks: List[mp.solutions.pose.PoseLandmark]) -> Dict[str, Optional[float]]:
        """计算所有关节角度（度），基于三点向量夹角"""
        angles = {}
        for joint_name, (a_idx, b_idx, c_idx) in self.angle_joint_indices.items():
            # 获取三点的可见性
            a_vis = landmarks[a_idx].visibility
            b_vis = landmarks[b_idx].visibility
            c_vis = landmarks[c_idx].visibility
            
            # 任意一点可见性不足：角度设为None
            if a_vis < self.visibility_threshold or b_vis < self.visibility_threshold or c_vis < self.visibility_threshold:
                angles[joint_name] = None
                continue
            
            # 提取三点坐标（归一化）
            a = np.array([landmarks[a_idx].x, landmarks[a_idx].y])  # 近端点
            b = np.array([landmarks[b_idx].x, landmarks[b_idx].y])  # 关节中心点
            c = np.array([landmarks[c_idx].x, landmarks[c_idx].y])  # 远端点
            
            # 计算向量ab（近端→关节中心）和向量bc（关节中心→远端）
            vec_ab = a - b
            vec_bc = c - b
            
            # 向量长度为0（异常情况）：角度设为None
            mag_ab = np.linalg.norm(vec_ab)
            mag_bc = np.linalg.norm(vec_bc)
            if mag_ab < 1e-6 or mag_bc < 1e-6:
                angles[joint_name] = None
                continue
            
            # 点积公式计算夹角（弧度转度）
            dot_product = np.dot(vec_ab, vec_bc)
            cos_theta = np.clip(dot_product / (mag_ab * mag_bc), -1.0, 1.0)  # 避免数值溢出
            theta_deg = np.degrees(np.arccos(cos_theta))
            angles[joint_name] = round(theta_deg, 2)  # 保留2位小数
        
        return angles

    def _calculate_joint_velocities(self, current_angles: Dict[str, Optional[float]], current_timestamp: float) -> Dict[str, Optional[float]]:
        """计算所有关节角速度（度/秒）"""
        velocities = {}
        # 无历史数据（第一帧或上一帧未检测到）：角速度设为None
        if not self.prev_joint_data:
            for joint_name in current_angles.keys():
                velocities[joint_name] = None
            return velocities
        
        # 计算时间差（当前帧 - 上一帧）
        delta_time = current_timestamp - self.prev_joint_data.timestamp
        if delta_time <= 0:  # 时间差无效（帧率异常）：角速度设为None
            for joint_name in current_angles.keys():
                velocities[joint_name] = None
            return velocities
        
        # 计算每个关节的角速度（|当前角度 - 上一帧角度| / 时间差）
        for joint_name, current_angle in current_angles.items():
            prev_angle = getattr(self.prev_joint_data, f"{joint_name}_angle", None)
            # 当前或上一帧角度无效：角速度设为None
            if current_angle is None or prev_angle is None:
                velocities[joint_name] = None
            else:
                delta_angle = abs(current_angle - prev_angle)
                velocities[joint_name] = round(delta_angle / delta_time, 2)  # 保留2位小数
        
        return velocities