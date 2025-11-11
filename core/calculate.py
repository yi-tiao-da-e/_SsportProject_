import cv2
import mediapipe as mp
import numpy as np

class PoseCalculator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 定义骨架连接关系
        self.POSE_CONNECTIONS = [
            # 身体连接
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            # 左臂
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            # 右臂
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            # 左腿
            (23, 25), (25, 27), (27, 29), (29, 31),
            # 右腿
            (24, 26), (26, 28), (28, 30), (30, 32),
            # 面部 (简化)
            (0, 1), (1, 2), (2, 3), (3, 7), 
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10)
        ]
    
    def process_video(self, video_path, progress_callback=None):
        """处理视频并返回3D关键点数据"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("无法打开视频文件")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        landmarks_3d = []
        frame_count = 0
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    x = landmark.x - 0.5
                    y = landmark.y - 0.5
                    z = landmark.z
                    landmarks.append([x, y, z, landmark.visibility])
                
                landmarks = self.adjust_pose(landmarks)
                landmarks_3d.append(landmarks)
            else:
                landmarks_3d.append([])
            
            frame_count += 1
            if progress_callback:
                progress_callback(frame_count, total_frames)
        
        cap.release()
        return landmarks_3d, total_frames
    
    def adjust_pose(self, landmarks):
        """调整姿态，使人体站立在地面上"""
        if not landmarks:
            return landmarks
        
        # 找到脚部最低点
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_heel = landmarks[29]
        right_heel = landmarks[30]
        
        foot_y_values = [left_ankle[1], right_ankle[1], left_heel[1], right_heel[1]]
        max_foot_y = max(foot_y_values)
        
        # 使脚部接触地面
        for i in range(len(landmarks)):
            landmarks[i][1] -= max_foot_y
        
        # 计算髋部中心点
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        hip_center = [
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2,
            (left_hip[2] + right_hip[2]) / 2
        ]
        
        # 将髋部中心点移到原点
        for i in range(len(landmarks)):
            landmarks[i][0] -= hip_center[0]
            landmarks[i][2] -= hip_center[2]
        
        return landmarks
    
    @staticmethod
    def rotate_point_y(point, angle_y):
        """绕Y轴旋转点"""
        x, y, z = point
        cos_y = np.cos(angle_y)
        sin_y = np.sin(angle_y)
        x_rot = x * cos_y + z * sin_y
        z_rot = -x * sin_y + z * cos_y
        return [x_rot, y, z_rot]
    
    def __del__(self):
        self.pose.close()