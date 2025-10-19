# recog/med_processor.py
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Set
from config.settings import app_config
# 从video_io的data_classes导入FrameData（而不是input_handler！）
from video_io.data_classes import FrameData  # 关键！消除循环导入
from .data_classes import JointData  # 从本模块导入JointData

# 初始化MediaPipe工具
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class MediaPipeProcessor:
    def __init__(self, selected_joints: Set[str] = None):
        # MediaPipe模型配置
        self.model_complexity = app_config.mediapipe.model_complexity
        self.min_detection_confidence = app_config.mediapipe.min_detection_confidence
        self.min_tracking_confidence = app_config.mediapipe.min_tracking_confidence
    
        # 选择的关节部位（None表示全选）
        self.selected_joints = selected_joints
        
        # 扩展关节关键点索引映射
        self.joint_index_map = {
            # 上半身
            "left_shoulder": 11, "left_elbow": 13, "left_wrist": 15,
            "right_shoulder": 12, "right_elbow": 14, "right_wrist": 16,
            # 下半身
            "left_hip": 23, "left_knee": 25, "left_ankle": 27,
            "right_hip": 24, "right_knee": 26, "right_ankle": 28
        }
        
        # 扩展角度计算所需的关节三点索引
        self.angle_joint_indices = {
            # 原有角度
            "left_elbow": (11, 13, 15),   # 左肩→左肘→左腕
            "right_elbow": (12, 14, 16),  # 右肩→右肘→右腕
            "left_knee": (23, 25, 27),    # 左髋→左膝→左踝
            "right_knee": (24, 26, 28),   # 右髋→右膝→右踝
            # 新增角度
            "left_hip": (11, 23, 25),     # 左肩→左髋→左膝（髋部角度）
            "right_hip": (12, 24, 26),    # 右肩→右髋→右膝
            "left_ankle": (23, 25, 27),   # 左髋→左膝→左踝（脚踝角度）
            "right_ankle": (24, 26, 28),  # 右髋→右膝→右踝
            "spine": ("neck", "hip_center")  # 脊柱角度（基于虚拟节点）
        }
        
        # 可见性阈值
        self.visibility_threshold = 0.5
        # 缓存上一帧JointData
        self.prev_joint_data: Optional[JointData] = None

    def process_frame(self, frame_data: FrameData) -> JointData:
        """处理一帧FrameData，返回结构化JointData"""
        rgb_frame = cv2.cvtColor(frame_data.frame, cv2.COLOR_BGR2RGB)
        
        with mp_pose.Pose(
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        ) as pose:            
            results = pose.process(rgb_frame)
        
        joint_data = JointData(
            frame_idx=frame_data.frame_idx,
            timestamp=frame_data.timestamp,
            fps=frame_data.fps
        )
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 提取关节坐标（带可见性过滤和部位选择过滤）
            joint_coords = self._extract_joint_coords(landmarks)
            for key, value in joint_coords.items():
                setattr(joint_data, key, value)
            
            # 计算虚拟节点坐标（颈椎和髋部中心）- 只在选择了相关部位时计算
            if self._should_process_virtual_points():
                self._calculate_virtual_points(joint_data, landmarks)
            
            # 计算关节角度 - 根据选择的部位过滤
            joint_angles = self._calculate_joint_angles(landmarks, joint_data)
            for key, value in joint_angles.items():
                setattr(joint_data, f"{key}_angle", value)
            
            # 计算关节角速度 - 根据选择的部位过滤
            joint_velocities = self._calculate_joint_velocities(joint_angles, frame_data.timestamp)
            for key, value in joint_velocities.items():
                setattr(joint_data, f"{key}_velocity", value)
            
            # 缓存当前关节数据
            self.prev_joint_data = joint_data
        else:
            self.prev_joint_data = None
        
        return joint_data

    def _should_process_joint(self, joint_name: str) -> bool:
        """检查是否应该处理指定关节"""
        if self.selected_joints is None:
            return True  # 未选择部位时处理所有部位
        return joint_name in self.selected_joints

    def _should_process_virtual_points(self) -> bool:
        """检查是否应该处理虚拟节点（颈椎和髋部中心）"""
        if self.selected_joints is None:
            return True
        # 如果选择了任何需要虚拟节点的部位，则处理虚拟节点
        virtual_related_joints = {"neck", "hip_center", "spine"}
        return any(joint in self.selected_joints for joint in virtual_related_joints)

    def _calculate_virtual_points(self, joint_data: JointData, landmarks: List[mp.solutions.pose.PoseLandmark]):
        """计算虚拟节点坐标（颈椎和髋部中心）"""
        # 计算颈椎虚拟节点（双肩中点）
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        neck_x = (left_shoulder.x + right_shoulder.x) / 2
        neck_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # 计算髋部中心虚拟节点（双髋中点）
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        hip_center_x = (left_hip.x + right_hip.x) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        
        # 设置虚拟节点坐标（只在选择了相关部位时设置）
        if self._should_process_joint("neck"):
            joint_data.neck_x = round(neck_x, 4)
            joint_data.neck_y = round(neck_y, 4)
        
        if self._should_process_joint("hip_center"):
            joint_data.hip_center_x = round(hip_center_x, 4)
            joint_data.hip_center_y = round(hip_center_y, 4)

    def _extract_joint_coords(self, landmarks: List[mp.solutions.pose.PoseLandmark]) -> Dict[str, Optional[float]]:
        """提取关节点坐标（带可见性过滤和部位选择过滤）"""
        coords = {}
        for joint_name, idx in self.joint_index_map.items():
            # 检查是否应该处理这个关节
            if not self._should_process_joint(joint_name):
                continue
                
            landmark = landmarks[idx]
            if landmark.visibility >= self.visibility_threshold:
                coords[f"{joint_name}_x"] = round(landmark.x, 4)
                coords[f"{joint_name}_y"] = round(landmark.y, 4)
            else:
                coords[f"{joint_name}_x"] = None
                coords[f"{joint_name}_y"] = None
        return coords

    def _calculate_joint_angles(self, landmarks: List[mp.solutions.pose.PoseLandmark], joint_data: JointData) -> Dict[str, Optional[float]]:
        """计算关节角度（度），包括脊柱角度，根据选择的部位过滤"""
        angles = {}
        
        for joint_name, indices in self.angle_joint_indices.items():
            # 检查是否应该处理这个关节的角度
            if not self._should_process_joint(joint_name):
                continue
                
            # 特殊处理脊柱角度（基于虚拟节点）
            if joint_name == "spine":
                spine_angle = self._calculate_spine_angle(joint_data)
                angles[joint_name] = spine_angle
                continue
                
            # 跳过虚拟节点的角度计算
            if isinstance(indices[0], str):
                continue
                
            a_idx, b_idx, c_idx = indices
            a = landmarks[a_idx]
            b = landmarks[b_idx]
            c = landmarks[c_idx]
            
            if (a.visibility < self.visibility_threshold or
                b.visibility < self.visibility_threshold or
                c.visibility < self.visibility_threshold):
                angles[joint_name] = None
                continue
            
            vec_ab = np.array([a.x - b.x, a.y - b.y])
            vec_cb = np.array([c.x - b.x, c.y - b.y])
            
            mag_ab = np.linalg.norm(vec_ab)
            mag_cb = np.linalg.norm(vec_cb)
            if mag_ab < 1e-6 or mag_cb < 1e-6:
                angles[joint_name] = None
                continue
            
            dot_product = np.dot(vec_ab, vec_cb)
            cos_theta = np.clip(dot_product / (mag_ab * mag_cb), -1.0, 1.0)
            theta_deg = np.degrees(np.arccos(cos_theta))
            angles[joint_name] = round(theta_deg, 2)
        
        return angles

    def _calculate_spine_angle(self, joint_data: JointData) -> Optional[float]:
        """计算脊柱角度（基于颈椎和髋部中心虚拟节点）"""
        if (joint_data.neck_x is None or joint_data.neck_y is None or
            joint_data.hip_center_x is None or joint_data.hip_center_y is None):
            return None
        
        # 计算脊柱向量（从髋部中心到颈椎）
        spine_vector = np.array([
            joint_data.neck_x - joint_data.hip_center_x,
            joint_data.neck_y - joint_data.hip_center_y
        ])
        
        # 计算与垂直方向的夹角
        vertical_vector = np.array([0, -1])  # 垂直向上
        
        mag_spine = np.linalg.norm(spine_vector)
        mag_vertical = np.linalg.norm(vertical_vector)
        
        if mag_spine < 1e-6:
            return None
        
        dot_product = np.dot(spine_vector, vertical_vector)
        cos_theta = np.clip(dot_product / (mag_spine * mag_vertical), -1.0, 1.0)
        theta_deg = np.degrees(np.arccos(cos_theta))
        
        return round(theta_deg, 2)

    def _calculate_joint_velocities(self, current_angles: Dict[str, Optional[float]], current_timestamp: float) -> Dict[str, Optional[float]]:
        """计算关节角速度（度/秒），根据选择的部位过滤"""
        velocities = {}
        if not self.prev_joint_data:
            for joint_name in current_angles.keys():
                velocities[joint_name] = None
            return velocities
        
        delta_time = current_timestamp - self.prev_joint_data.timestamp
        if delta_time <= 0:
            for joint_name in current_angles.keys():
                velocities[joint_name] = None
            return velocities
        
        for joint_name, current_angle in current_angles.items():
            # 检查是否应该处理这个关节的角速度
            if not self._should_process_joint(joint_name):
                continue
                
            prev_angle = getattr(self.prev_joint_data, f"{joint_name}_angle", None)
            if current_angle is None or prev_angle is None:
                velocities[joint_name] = None
            else:
                delta_angle = abs(current_angle - prev_angle)
                velocities[joint_name] = round(delta_angle / delta_time, 2)
        
        return velocities