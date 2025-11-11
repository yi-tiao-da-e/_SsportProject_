import tkinter as tk
from tkinter import ttk, filedialog
from video_io.input import VideoInput
from video_io.output import VideoOutput
from core.calculate import PoseCalculator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import threading
import numpy as np
import gc
import time
from core.smplx_model import SMPLXModelHandler  # 导入SMPLX处理器
import cv2

class PoseDetectionApp:
    def __init__(self, root):
        self.root = root
        root.title("3D姿态检测与分析系统")
        root.geometry("1200x800")
        
        # 初始化组件
        self.video_input = VideoInput()
        self.video_output = VideoOutput()
        self.pose_calculator = PoseCalculator()
        
        # 视频相关属性
        self.video_path = None
        self.landmarks_3d = []
        self.current_frame_idx = 0
        self.is_playing = False
        self.is_processing = False
        self.model_rotation_y = 0
        
        # SMPLX模型相关
        self.use_smplx = False  # 默认不使用
        self.smplx_var = tk.BooleanVar()
        self.smplx_handler = None
        self.smplx_joints = []  # 存储SMPLX关节数据
        self.smplx_connections = []  # 骨架连接关系
        
        # 处理控制
        self.stop_processing = False
        self.processing_thread = None
        
        # 创建界面
        self.create_widgets()
        
        # 状态变量
        self.status_var = tk.StringVar()
        self.status_var.set("请选择视频文件")
        
        # 进度变量
        self.progress_var = tk.DoubleVar()
        self.progress_var.set(0.0)
        
        # 初始化其他状态
        self.create_status_bar()

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
        
        # SMPLX开关
        self.smplx_var = tk.BooleanVar(value=self.use_smplx)
        smplx_check = ttk.Checkbutton(
            control_frame, 
            text="使用SMPLX模型", 
            variable=self.smplx_var,
            command=self.toggle_smplx
        )
        smplx_check.pack(side=tk.LEFT, padx=5)
        
        # 视角控制
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
        
        # 绑定鼠标事件
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
        
    def create_status_bar(self):
        pass  # 已在create_widgets中实现
    
    def toggle_smplx(self):
        """切换SMPLX模型使用状态"""
        self.use_smplx = self.smplx_var.get()
        if self.use_smplx:
            if self.smplx_handler is None:
                self.init_smplx_handler()
        else:
            # 清理SMPLX资源
            if self.smplx_handler:
                self.smplx_handler.cleanup()
                self.smplx_handler = None
                self.smplx_joints = []
                
        self.update_3d_display(self.current_frame_idx)
    
    def init_smplx_handler(self):
        """初始化SMPLX处理器"""
        try:
            # 尝试多个可能的路径
            possible_paths = [
                "./models/smplx",
                "F:/_SsportProject_/models/smplx",  # 您的项目路径
                "C:/models/smplx"
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path) and os.listdir(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError("未找到SMPLX模型文件，请确保模型位于models/smplx目录")
            
            self.status_var.set("正在加载SMPLX模型...")
            
            # 在后台线程中加载
            def load_in_background():
                try:
                    self.smplx_handler = SMPLXModelHandler(model_path)
                    self.smplx_handler.initialize()
                    self.smplx_connections = self.smplx_handler.get_connections()
                    self.root.after(0, lambda: self.status_var.set("SMPLX模型已加载"))
                except Exception as e:
                    self.root.after(0, lambda: self.status_var.set(f"加载SMPLX失败: {str(e)}"))
                    self.root.after(0, lambda: self.smplx_var.set(False))
                    self.use_smplx = False
            
            threading.Thread(target=load_in_background, daemon=True).start()
        except Exception as e:
            self.status_var.set(f"初始化SMPLX失败: {str(e)}")
            self.smplx_var.set(False)
            self.use_smplx = False
    
    def select_video(self):
        self.video_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        
        if self.video_path:
            self.process_btn.config(state=tk.NORMAL)
            self.status_var.set(f"已选择视频: {os.path.basename(self.video_path)}")
            if self.video_input.test_video_file(self.video_path):
                self.status_var.set(f"已选择视频: {os.path.basename(self.video_path)} (可读)")
            else:
                self.status_var.set(f"错误: 无法读取视频帧")
                self.process_btn.config(state=tk.DISABLED)
    
    def process_video(self):
        if not self.video_path:
            return
        
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.status_var.set("正在处理视频...")
        
        # 启动处理线程
        self.processing_thread = threading.Thread(
            target=self._process_frames_thread,
            daemon=True
        )
        self.processing_thread.start()
    
    def _process_frames_thread(self):
        try:
            # 初始化视频输入
            video_input = self.video_input.open_video(self.video_path)
            if not video_input.isOpened():
                self.root.after(0, lambda: self.status_var.set("无法打开视频文件"))
                return
                
            # 获取视频信息
            total_frames = int(video_input.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video_input.get(cv2.CAP_PROP_FPS)
            
            self.landmarks_3d = []
            frame_idx = 0
            
            while not self.stop_processing and frame_idx < total_frames:
                ret, frame = video_input.read()
                if not ret:
                    break
                
                # 更新进度
                progress = (frame_idx / total_frames) * 100
                self.root.after(0, lambda: self.progress_var.set(progress))
                self.root.after(0, lambda: self.status_var.set(f"处理中: {frame_idx}/{total_frames}帧"))
                
                # 处理当前帧
                results = self.pose_calculator.process_frame(frame)
                landmarks_3d = self.pose_calculator.get_3d_landmarks(results).astype(np.float16)  # 使用低精度
                self.landmarks_3d.append(landmarks_3d)
                
                # 定期释放资源
                if frame_idx % 50 == 0:
                    gc.collect()
                
                frame_idx += 1
                
                # 添加延迟防止UI冻结
                time.sleep(0.01)
            
            video_input.release()
            
            # 处理完成
            self.root.after(0, lambda: self._processing_complete(total_frames))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"处理错误: {str(e)}"))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.is_processing = False
    
    def _processing_complete(self, total_frames):
        self.is_processing = False
        self.play_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.process_btn.config(state=tk.NORMAL)
        
        # 更新帧控制
        self.frame_scale.config(to=len(self.landmarks_3d)-1)
        self.frame_label.config(text=f"帧: 0/{len(self.landmarks_3d)-1}")
        self.status_var.set(f"处理完成! 共处理 {len(self.landmarks_3d)} 帧")
        self.update_3d_display(0)
        
        # 如果启用了SMPLX模型，则进行拟合
        if self.use_smplx and self.smplx_handler:
            self.status_var.set("正在拟合SMPLX模型...")
            
            # 在后台线程中拟合
            def fit_smplx():
                try:
                    # 使用低精度数据
                    landmarks_array = np.array(self.landmarks_3d, dtype=np.float16)
                    self.smplx_joints = self.smplx_handler.fit_to_keypoints(landmarks_array)
                    self.root.after(0, lambda: self.status_var.set("SMPLX拟合完成!"))
                    self.root.after(0, lambda: self.update_3d_display(self.current_frame_idx))
                except Exception as e:
                    self.root.after(0, lambda: self.status_var.set(f"SMPLX拟合错误: {str(e)}"))
                    self.smplx_joints = []
            
            threading.Thread(target=fit_smplx, daemon=True).start()
        else:
            self.smplx_joints = []
            self.status_var.set("处理完成!")
    
    def toggle_play(self):
        if not self.landmarks_3d:
            return
            
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.config(text="暂停")
            self.play_animation()
        else:
            self.play_btn.config(text="播放")
    
    def play_animation(self):
        if not self.is_playing or not self.landmarks_3d:
            return
        
        self.current_frame_idx += 1
        if self.current_frame_idx >= len(self.landmarks_3d):
            self.current_frame_idx = 0
        
        self.update_3d_display(self.current_frame_idx)
        self.frame_var.set(self.current_frame_idx)
        self.frame_label.config(text=f"帧: {self.current_frame_idx}/{len(self.landmarks_3d)-1}")
        
        if self.is_playing:
            self.root.after(33, self.play_animation)  # 约30 FPS
    
    def on_frame_change(self, value):
        if not self.is_playing:
            frame_idx = int(float(value))
            self.current_frame_idx = frame_idx
            self.update_3d_display(frame_idx)
            self.frame_label.config(text=f"帧: {frame_idx}/{len(self.landmarks_3d)-1}")
    
    def update_3d_display(self, frame_idx):
        """更新3D显示"""
        self.ax.clear()
        
        # 设置坐标轴和视角
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.view_init(elev=20, azim=120)
        
        # 检查是否有可用数据
        if not self.landmarks_3d or frame_idx >= len(self.landmarks_3d):
            self.ax.text(0, 0, 0, "没有姿态数据", ha='center')
            self.canvas.draw()
            return
            
        # 根据是否使用SMPLX选择数据源
        if self.use_smplx and len(self.smplx_joints) > 0 and frame_idx < len(self.smplx_joints):
            self._draw_smplx_joints(self.smplx_joints[frame_idx])
        else:
            landmarks = self.landmarks_3d[frame_idx]
            self._draw_mediapipe_landmarks(landmarks)
        
        self.canvas.draw()
    
    def _draw_smplx_joints(self, joints: np.ndarray):
        """绘制SMPLX关节"""
        # 旋转点以适应视角
        rotated_points = []
        for point in joints:
            rotated_point = self._rotate_point_y(point, np.radians(self.model_rotation_y))
            rotated_points.append(rotated_point)
        
        rotated_points = np.array(rotated_points)
        
        # 提取坐标
        xs = rotated_points[:, 0]
        ys = rotated_points[:, 1]
        zs = rotated_points[:, 2]
        
        # 绘制关节点
        self.ax.scatter(xs, ys, zs, c='g', marker='o', s=20)
        
        # 绘制连接线
        for start_idx, end_idx in self.smplx_connections:
            if start_idx < len(joints) and end_idx < len(joints):
                start = rotated_points[start_idx]
                end = rotated_points[end_idx]
                self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'b-', linewidth=2)
        
        # 添加阴影和标题
        self._draw_shadows()
        self.ax.set_title(f'SMPLX 3D姿态 - 帧 {self.current_frame_idx}')
    
    def _draw_mediapipe_landmarks(self, landmarks):
        """绘制MediaPipe的3D关键点"""
        # 旋转点以适应视角
        rotated_landmarks = []
        for landmark in landmarks:
            rotated_point = self._rotate_point_y(
                [landmark[0], landmark[1], landmark[2]], 
                np.radians(self.model_rotation_y)
            )
            rotated_landmarks.append(np.append(rotated_point, landmark[3]))
        
        rotated_landmarks = np.array(rotated_landmarks)
        
        # 提取坐标
        xs = rotated_landmarks[:, 0]
        ys = rotated_landmarks[:, 1]
        zs = rotated_landmarks[:, 2]
        
        # 绘制关节点
        self.ax.scatter(xs, ys, zs, c='r', marker='o', s=20)
        
        # 绘制连接线
        connections = self.pose_calculator.POSE_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            if (start_idx < len(rotated_landmarks) and end_idx < len(rotated_landmarks) and
                rotated_landmarks[start_idx][3] > 0.5 and rotated_landmarks[end_idx][3] > 0.5):
                start = rotated_landmarks[start_idx]
                end = rotated_landmarks[end_idx]
                self.ax.plot(
                    [start[0], end[0]], 
                    [start[1], end[1]], 
                    [start[2], end[2]], 
                    'b-', 
                    linewidth=2
                )
        
        # 添加阴影和标题
        self._draw_shadows()
        self.ax.set_title(f'MediaPipe 3D姿态 - 帧 {self.current_frame_idx}')
    
    def _rotate_point_y(self, point, angle):
        """绕Y轴旋转点"""
        x, y, z = point
        rotated_x = x * np.cos(angle) + z * np.sin(angle)
        rotated_z = -x * np.sin(angle) + z * np.cos(angle)
        return [rotated_x, y, rotated_z]
    
    def _draw_shadows(self):
        """绘制地面和墙壁作为参考"""
        # 绘制地面
        x = np.arange(-0.5, 0.5, 0.1)
        z = np.arange(-0.5, 0.5, 0.1)
        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X)
        self.ax.plot_surface(X, Y, Z, alpha=0.1, color='gray', shade=True)
        
        # 绘制后墙
        y = np.arange(-1.0, 0, 0.1)
        z = np.arange(-0.5, 0.5, 0.1)
        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, -0.5)
        self.ax.plot_surface(X, Y, Z, alpha=0.05, color='gray', shade=True)
        
        # 绘制侧墙
        x = np.arange(-0.5, 0.5, 0.1)
        y = np.arange(-1.0, 0, 0.1)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, -0.5)
        self.ax.plot_surface(X, Y, Z, alpha=0.05, color='gray', shade=True)
    
    def reset_view(self):
        """重置视角"""
        self.model_rotation_y = 0
        self.update_3d_display(self.current_frame_idx)
    
    def on_click(self, event):
        """鼠标点击事件"""
        if event.inaxes == self.ax:
            self.dragging = True
            self.last_mouse_pos = (event.x, event.y)
    
    def on_motion(self, event):
        """鼠标移动事件"""
        if self.dragging and event.inaxes == self.ax and self.last_mouse_pos:
            dx = event.x - self.last_mouse_pos[0]
            self.model_rotation_y -= dx * 0.5
            self.update_3d_display(self.current_frame_idx)
            self.last_mouse_pos = (event.x, event.y)
    
    def on_release(self, event):
        """鼠标释放事件"""
        self.dragging = False
        self.last_mouse_pos = None
    
    def save_data(self):
        """保存姿态数据"""
        if not self.landmarks_3d:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy文件", "*.npy"), ("所有文件", "*.*")]
        )
        
        if file_path:
            landmarks_array = np.array(self.landmarks_3d, dtype=np.float16)  # 使用低精度存储
            np.save(file_path, landmarks_array)
            self.status_var.set(f"姿态数据已保存到: {os.path.basename(file_path)}")
    
    def __del__(self):
        """清理资源"""
        if self.smplx_handler:
            self.smplx_handler.cleanup()


class VideoInput:
    """视频输入处理类"""
    def select_video_file(self):
        return filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
    
    def test_video_file(self, path):
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret
            return False
        except:
            return False
    
    def open_video(self, path):
        import cv2
        return cv2.VideoCapture(path)


class VideoOutput:
    """视频输出处理类"""
    def save_landmarks_data(self, landmarks):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy文件", "*.npy"), ("所有文件", "*.*")]
        )
        
        if file_path:
            np.save(file_path, np.array(landmarks, dtype=np.float16))  # 使用低精度存储
            return file_path
        return None


class PoseCalculator:
    """姿态计算类"""
    # 使用MediaPipe Pose的连接关系
    POSE_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
        (11, 12), (11, 13), (13, 15),  # 左臂
        (12, 14), (14, 16),  # 右臂
        (11, 23), (12, 24),  # 肩膀到臀部
        (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),  # 腿部
        (27, 31), (28, 32)  # 脚踝到脚尖
    ]
    
    def __init__(self):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose.Pose(
            model_complexity=1,  # 使用低复杂度模型
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False,
            smooth_landmarks=True
        )
    
    def process_frame(self, frame):
        """处理视频帧并返回结果"""
        import cv2
        # 获取图像尺寸
        height, width = frame.shape[:2]
        
        # 转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 创建媒体pipe的图像对象（包含尺寸信息）
        from mediapipe.framework.formats import image_frame
        mp_image = image_frame.ImageFrame(
            image_format=image_frame.ImageFormat.SRGB,
            data=rgb_frame,
            width=width,
            height=height
        )
        
        # 处理帧
        results = self.mp_pose.process(mp_image)
        return results
    
    def get_3d_landmarks(self, results):
        """从结果中提取3D关键点"""
        landmarks = []
        if results.pose_world_landmarks:
            for landmark in results.pose_world_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            return np.array(landmarks, dtype=np.float16)  # 使用低精度
        return np.zeros((33, 4), dtype=np.float16)


if __name__ == "__main__":
    root = tk.Tk()
    app = PoseDetectionApp(root)
    root.mainloop()