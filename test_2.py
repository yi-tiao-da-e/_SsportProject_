import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
import os
import pickle

class PoseDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D姿态检测")
        self.root.geometry("1200x800")
        
        # 初始化变量
        self.video_path = None
        self.is_processing = False
        self.is_playing = False
        self.landmarks_3d = []
        self.current_frame_idx = 0
        self.total_frames = 0
        
        # 模型旋转控制变量 - 只保留Y轴旋转
        self.model_rotation_y = 0  # 绕Y轴旋转角度
        
        # 初始化MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 创建UI
        self.create_widgets()
        
    def create_widgets(self):
        # 控制面板
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        self.select_btn = tk.Button(control_frame, text="选择视频", command=self.select_video)
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        self.process_btn = tk.Button(control_frame, text="处理视频", command=self.process_video, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = tk.Button(control_frame, text="播放", command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = tk.Button(control_frame, text="保存数据", command=self.save_data, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # 只保留重置视角按钮
        view_frame = tk.Frame(self.root)
        view_frame.pack(pady=5)
        
        self.reset_view_btn = tk.Button(view_frame, text="重置视角", command=self.reset_view)
        self.reset_view_btn.pack(side=tk.LEFT, padx=5)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # 帧控制
        frame_control = tk.Frame(self.root)
        frame_control.pack(pady=5)
        
        self.frame_var = tk.IntVar()
        self.frame_scale = tk.Scale(frame_control, from_=0, to=100, orient=tk.HORIZONTAL, 
                                   variable=self.frame_var, command=self.on_frame_change, 
                                   length=400, showvalue=True)
        self.frame_scale.pack(side=tk.LEFT, padx=5)
        
        self.frame_label = tk.Label(frame_control, text="帧: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        
        # 3D骨架显示
        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 绑定鼠标事件用于视角控制
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("请选择视频文件")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 鼠标拖动状态
        self.dragging = False
        self.last_mouse_pos = None
        
    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.video_path = file_path
            self.process_btn.config(state=tk.NORMAL)
            self.status_var.set(f"已选择视频: {os.path.basename(file_path)}")
            
            # 测试视频文件是否可以打开
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.status_var.set(f"已选择视频: {os.path.basename(file_path)} (可读)")
                else:
                    self.status_var.set(f"错误: 无法读取视频帧")
                    self.process_btn.config(state=tk.DISABLED)
                cap.release()
            else:
                self.status_var.set(f"错误: 无法打开视频文件")
                self.process_btn.config(state=tk.DISABLED)
    
    def process_video(self):
        if not self.video_path:
            return
        
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.status_var.set("正在处理视频...")
        
        # 启动处理线程
        thread = threading.Thread(target=self.process_frames)
        thread.daemon = True
        thread.start()
    
    def process_frames(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.root.after(0, lambda: self.status_var.set("错误: 无法打开视频文件"))
                return
            
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.total_frames == 0:
                self.root.after(0, lambda: self.status_var.set("错误: 视频文件为空或无法读取"))
                cap.release()
                return
            
            frame_count = 0
            self.landmarks_3d = []
            
            while self.is_processing and frame_count < self.total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 进行姿态检测
                results = self.pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # 提取3D关键点并调整坐标系
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        # 调整坐标系，使模型正立
                        # MediaPipe的坐标系原点在图像中心，y轴向下，我们需要调整
                        # 同时调整坐标范围，使人体居中
                        x = landmark.x - 0.5
                        y = landmark.y - 0.5
                        z = landmark.z
                        landmarks.append([x, y, z, landmark.visibility])
                    
                    # 调整姿态，使人体站立在地面上
                    landmarks = self.adjust_pose(landmarks)
                    self.landmarks_3d.append(landmarks)
                else:
                    # 如果没有检测到姿态，添加空列表
                    self.landmarks_3d.append([])
                
                frame_count += 1
                
                # 更新进度
                progress = (frame_count / self.total_frames) * 100
                self.root.after(0, lambda: self.progress_var.set(progress))
                self.root.after(0, lambda: self.status_var.set(f"处理进度: {frame_count}/{self.total_frames} 帧"))
            
            cap.release()
            
            # 处理完成
            self.is_processing = False
            self.root.after(0, self.processing_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"处理错误: {str(e)}"))
            self.is_processing = False
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
    
    def adjust_pose(self, landmarks):
        """调整姿态，使人体站立在地面上"""
        if not landmarks:
            return landmarks
        
        # 找到脚部关键点（脚踝和脚跟）
        left_ankle = landmarks[27]  # 左脚踝
        right_ankle = landmarks[28]  # 右脚踝
        left_heel = landmarks[29]    # 左脚跟
        right_heel = landmarks[30]   # 右脚跟
        
        # 计算脚部最低点（Y值最大，因为MediaPipe的Y轴向下为正）
        foot_y_values = [left_ankle[1], right_ankle[1], left_heel[1], right_heel[1]]
        max_foot_y = max(foot_y_values)  # 注意：这里是最大值，因为Y轴向下为正
        
        # 将整个身体下移，使脚部接触地面（Y=0）
        # 由于Y轴向下为正，我们需要减去最大值，使脚部在Y=0位置
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
    
    def rotate_point(self, point, angle_y):
        """绕Y轴旋转点"""
        x, y, z = point
        
        # 绕Y轴旋转
        cos_y = np.cos(angle_y)
        sin_y = np.sin(angle_y)
        x_rot = x * cos_y + z * sin_y
        z_rot = -x * sin_y + z * cos_y
        x, z = x_rot, z_rot
        
        return [x, y, z]
    
    def processing_complete(self):
        self.play_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.frame_scale.config(to=len(self.landmarks_3d)-1)
        self.frame_label.config(text=f"帧: 0/{len(self.landmarks_3d)-1}")
        self.status_var.set(f"处理完成! 共处理 {len(self.landmarks_3d)} 帧")
        
        # 显示第一帧
        self.update_3d_display(0)
    
    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.config(text="暂停")
            self.play_animation()
        else:
            self.play_btn.config(text="播放")
    
    def play_animation(self):
        if not self.is_playing:
            return
        
        # 更新当前帧
        self.current_frame_idx += 1
        if self.current_frame_idx >= len(self.landmarks_3d):
            self.current_frame_idx = 0
        
        # 更新显示
        self.update_3d_display(self.current_frame_idx)
        
        # 更新UI
        self.frame_var.set(self.current_frame_idx)
        self.frame_label.config(text=f"帧: {self.current_frame_idx}/{len(self.landmarks_3d)-1}")
        
        # 安排下一帧
        if self.is_playing:
            self.root.after(33, self.play_animation)  # 约30fps
    
    def on_frame_change(self, value):
        if not self.is_playing:
            frame_idx = int(float(value))
            self.current_frame_idx = frame_idx
            self.update_3d_display(frame_idx)
            self.frame_label.config(text=f"帧: {frame_idx}/{len(self.landmarks_3d)-1}")
    
    def update_3d_display(self, frame_idx):
        if not self.landmarks_3d or frame_idx >= len(self.landmarks_3d):
            return
        
        # 清除之前的绘图
        self.ax.clear()
        
        # 调整坐标轴范围，使人站在地板上
        self.ax.set_xlim3d(-0.5, 0.5)
        self.ax.set_ylim3d(-1.0, 0)  # Y轴从-1到0，地板在Y=0
        self.ax.set_zlim3d(-0.5, 0.5)
        
        # 隐藏坐标轴
        self.ax.set_axis_off()
        
        # 绘制阴影（地面和墙面）
        self.draw_shadows()
        
        # 固定视角 - 正面视角
        self.ax.view_init(elev=-80, azim=-90)
        
        # 获取当前帧的关节点
        landmarks = self.landmarks_3d[frame_idx]
        
        if not landmarks:
            # 如果没有检测到姿态，显示提示
            self.ax.text(0, -0.5, 0, "未检测到姿态", fontsize=12, ha='center')
        else:
            # 应用模型旋转（只绕Y轴）
            rotated_landmarks = []
            for landmark in landmarks:
                # 将角度转换为弧度
                angle_y_rad = np.radians(self.model_rotation_y)
                
                # 旋转点（只绕Y轴）
                rotated_point = self.rotate_point([landmark[0], landmark[1], landmark[2]], angle_y_rad)
                rotated_landmarks.append(rotated_point + [landmark[3]])  # 保留可见性
            
            # 绘制骨架连接线
            connections = [
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
            
            # 绘制关节点
            xs, ys, zs = [], [], []
            for landmark in rotated_landmarks:
                xs.append(landmark[0])
                ys.append(landmark[1])
                zs.append(landmark[2])
            
            self.ax.scatter(xs, ys, zs, c='r', marker='o', s=20)
            
            # 绘制连接线
            for connection in connections:
                start_idx, end_idx = connection
                if (start_idx < len(rotated_landmarks) and end_idx < len(rotated_landmarks) and
                    rotated_landmarks[start_idx][3] > 0.5 and rotated_landmarks[end_idx][3] > 0.5):  # 可见性检查
                    start_point = rotated_landmarks[start_idx]
                    end_point = rotated_landmarks[end_idx]
                    self.ax.plot([start_point[0], end_point[0]], 
                                [start_point[1], end_point[1]], 
                                [start_point[2], end_point[2]], 'b-', linewidth=2)
        
        # 设置标题
        self.ax.set_title(f'3D姿态 - 帧 {frame_idx}')
        
        # 刷新画布
        self.canvas.draw()
    
    def draw_shadows(self):
        """绘制地面和墙面阴影"""
        # 创建地面网格 - 地板在Y=0
        x = np.arange(-0.5, 0.5, 0.1)
        z = np.arange(-0.5, 0.5, 0.1)
        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X)  # 地面在Y=0
        
        # 绘制地面网格 - 使用更淡的颜色
        self.ax.plot_surface(X, Y, Z, alpha=0.1, color='gray', shade=True)
        
        # 创建后墙网格
        y = np.arange(-1.0, 0, 0.1)
        z = np.arange(-0.5, 0.5, 0.1)
        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, -0.5)  # 后墙在X=-0.5
        
        # 绘制后墙网格 - 使用更淡的颜色
        self.ax.plot_surface(X, Y, Z, alpha=0.05, color='gray', shade=True)
        
        # 创建侧墙网格
        x = np.arange(-0.5, 0.5, 0.1)
        y = np.arange(-1.0, 0, 0.1)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, -0.5)  # 侧墙在Z=-0.5
        
        # 绘制侧墙网格 - 使用更淡的颜色
        self.ax.plot_surface(X, Y, Z, alpha=0.05, color='gray', shade=True)
    
    def reset_view(self):
        # 重置模型旋转角度
        self.model_rotation_y = 0
        self.update_3d_display(self.current_frame_idx)
    
    def on_click(self, event):
        if event.inaxes == self.ax:
            self.dragging = True
            self.last_mouse_pos = (event.x, event.y)
    
    def on_motion(self, event):
        if self.dragging and event.inaxes == self.ax and self.last_mouse_pos:
            dx = event.x - self.last_mouse_pos[0]
            # 忽略dy（上下移动）
            
            # 只响应左右移动，绕Y轴旋转模型（旋转中心在髋部）
            self.model_rotation_y -= dx * 0.5  # 左右移动绕Y轴旋转模型
            
            # 更新显示
            self.update_3d_display(self.current_frame_idx)
            
            self.last_mouse_pos = (event.x, event.y)
    
    def on_release(self, event):
        self.dragging = False
        self.last_mouse_pos = None
    
    def save_data(self):
        if not self.landmarks_3d:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存姿态数据",
            defaultextension=".pkl",
            filetypes=[("Pickle文件", "*.pkl"), ("所有文件", "*.*")]
        )
        
        if file_path:
            # 保存姿态数据
            with open(file_path, 'wb') as f:
                pickle.dump(self.landmarks_3d, f)
            
            self.status_var.set(f"姿态数据已保存到: {os.path.basename(file_path)}")

    def __del__(self):
        # 确保释放资源
        if hasattr(self, 'pose'):
            self.pose.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = PoseDetectionApp(root)
    
    try:
        root.mainloop()
    finally:
        # 确保应用关闭时释放资源
        if hasattr(app, 'pose'):
            app.pose.close()