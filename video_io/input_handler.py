from typing import Iterator, List, Dict  # 根据需要调整，确保有必要的类型提示
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import time
import os  # 新增os导入
from dataclasses import dataclass
from config.settings import app_config
from recog.med_processor import MediaPipeProcessor
from video_io.presentation import Presentation
from video_io.data_classes import FrameData  # 确保FrameData导入正确
@dataclass
class FrameData:
    frame: cv2.Mat
    timestamp: float
    fps: float
    frame_idx: int

class InputHandler:
    def __init__(self, master):  # 新增：接收主窗口master
        self.master = master  # 保存主窗口实例（用于创建Toplevel）
        self.input_source = None
        self.input_type = None
        self.processing = False
        self.cap = None
        self.selected_joints = []  # 初始化选中关节列表
        
    def show_input_dialog(self) -> bool:
        """显示输入源与关节选择弹窗（模态，依赖主窗口）"""
        # 创建Toplevel子窗口（主窗口的子窗口，共享事件循环）
        dialog = tk.Toplevel(self.master)  # 关键：使用self.master作为父窗口
        dialog.title("运动分析设置")
        dialog.geometry("850x650")
        dialog.resizable(False, False)
        dialog.attributes("-topmost", True)  # 窗口置顶（优先显示）

        # ------------------------------
        # 输入源选择（保持原逻辑，调整Label更新方式）
        # ------------------------------
        input_frame = tk.Frame(dialog, padx=10, pady=10)
        input_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(input_frame, text="输入源选择", font=("Arial", 12, "bold")).pack(pady=5)

        def select_file():
            self.input_source = filedialog.askopenfilename(
                title="选择视频文件",
                filetypes=[("视频文件", "*.mp4 *.avi *.mov")],
                initialdir="./"
            )
            if self.input_source:
                self.input_type = "file"
                # 销毁已有的选中文件Label（避免重复显示）
                for widget in input_frame.winfo_children():
                    if isinstance(widget, tk.Label) and widget.cget("font") == ("Arial", 10, "italic"):
                        widget.destroy()
                # 添加新的选中文件Label
                tk.Label(input_frame, text=f"选中文件：{os.path.basename(self.input_source)}", font=("Arial", 10, "italic")).pack(pady=5)
            else:
                messagebox.showwarning("警告", "未选择文件")

        def select_camera():
            self.input_source = 0
            self.input_type = "camera"
            # 销毁已有的选中摄像头Label
            for widget in input_frame.winfo_children():
                if isinstance(widget, tk.Label) and widget.cget("font") == ("Arial", 10, "italic"):
                    widget.destroy()
                # 添加新的选中摄像头Label
                tk.Label(input_frame, text="选中摄像头：默认摄像头（ID=0）", font=("Arial", 10, "italic")).pack(pady=5)

        tk.Button(input_frame, text="本地视频文件", command=select_file, width=20).pack(pady=5)
        tk.Button(input_frame, text="摄像头", command=select_camera, width=20).pack(pady=5)

        # ------------------------------
        # 预设方案选择（保持原逻辑）
        # ------------------------------
        preset_frame = tk.Frame(dialog, padx=10, pady=10)
        preset_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(preset_frame, text="预设方案", font=("Arial", 12, "bold")).pack(pady=5)

        var_preset = tk.StringVar(value="none")

        def apply_preset(preset: str):
            """应用预设方案（重置关节选择）"""
            # 重置所有关节为未选中（蓝色）
            for joint in joint_tags:
                canvas.itemconfig(joint, fill="blue")
                selected_joints_dict[joint] = False
            # 选中预设对应的关节（红色）
            target_joints = {
                "upper": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
                "lower": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
                "full": list(joint_tags.keys())
            }.get(preset, [])
            for joint in target_joints:
                canvas.itemconfig(joint, fill="red")
                selected_joints_dict[joint] = True
            # 刷新选中关节列表
            update_selected_joints()

        tk.Radiobutton(preset_frame, text="上肢（肩/肘/腕）", variable=var_preset, value="upper", command=lambda: apply_preset("upper")).pack(anchor=tk.W, pady=2)
        tk.Radiobutton(preset_frame, text="下肢（髋/膝/踝）", variable=var_preset, value="lower", command=lambda: apply_preset("lower")).pack(anchor=tk.W, pady=2)
        tk.Radiobutton(preset_frame, text="全身（所有关节）", variable=var_preset, value="full", command=lambda: apply_preset("full")).pack(anchor=tk.W, pady=2)

        # ------------------------------
        # 人体骨架与关节选择（保持原逻辑）
        # ------------------------------
        canvas = tk.Canvas(dialog, width=500, height=600, bg="white")
        canvas.pack(side=tk.RIGHT, padx=10, pady=10)

        def draw_skeleton():
            """绘制简化人体骨架"""
            canvas.create_oval(220, 20, 280, 80, fill="gray", outline="black")  # 头部
            canvas.create_line(250, 80, 250, 250, width=2)  # 躯干
            # 左臂（肩→肘→腕）
            canvas.create_line(250, 100, 180, 150, width=2)
            canvas.create_line(180, 150, 120, 200, width=2)
            # 右臂（肩→肘→腕）
            canvas.create_line(250, 100, 320, 150, width=2)
            canvas.create_line(320, 150, 380, 200, width=2)
            # 左腿（髋→膝→踝）
            canvas.create_line(250, 250, 180, 350, width=2)
            canvas.create_line(180, 350, 120, 450, width=2)
            # 右腿（髋→膝→踝）
            canvas.create_line(250, 250, 320, 350, width=2)
            canvas.create_line(320, 350, 380, 450, width=2)

        draw_skeleton()

        # 关节位置映射（键：关节名称，值：(x1, y1, x2, y2)）
        joint_tags = {
            "left_shoulder": (230, 90, 250, 110),   # 左肩
            "right_shoulder": (250, 90, 270, 110),  # 右肩
            "left_elbow": (170, 140, 190, 160),     # 左肘
            "right_elbow": (310, 140, 330, 160),    # 右肘
            "left_wrist": (110, 190, 130, 210),     # 左腕
            "right_wrist": (370, 190, 390, 210),    # 右腕
            "left_hip": (230, 240, 250, 260),       # 左髋
            "right_hip": (250, 240, 270, 260),      # 右髋
            "left_knee": (170, 340, 190, 360),      # 左膝
            "right_knee": (310, 340, 330, 360),     # 右膝
            "left_ankle": (110, 440, 130, 460),     # 左踝
            "right_ankle": (370, 440, 390, 460)     # 右踝
        }

        # 初始化关节状态（未选中：蓝色）
        selected_joints_dict = {joint: False for joint in joint_tags}
        for joint, coords in joint_tags.items():
            x1, y1, x2, y2 = coords
            canvas.create_oval(x1, y1, x2, y2, fill="blue", tags=joint)
            # 绑定点击事件（切换选中状态）
            canvas.tag_bind(joint, "<Button-1>", lambda e, j=joint: toggle_joint(j))

        def toggle_joint(joint: str):
            """切换关节选中状态（蓝色→红色/红色→蓝色）"""
            selected_joints_dict[joint] = not selected_joints_dict[joint]
            color = "red" if selected_joints_dict[joint] else "blue"
            canvas.itemconfig(joint, fill=color)
            # 刷新选中关节列表
            update_selected_joints()

        def update_selected_joints():
            """更新选中关节列表（格式：[{"side": "left", "part": "shoulder"}, ...]）"""
            self.selected_joints = []
            for joint in selected_joints_dict:
                if selected_joints_dict[joint]:
                    side, part = joint.split("_")
                    self.selected_joints.append({"side": side, "part": part})

        # ------------------------------
        # 操作按钮（保持原逻辑）
        # ------------------------------
        button_frame = tk.Frame(dialog, padx=10, pady=10)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        def on_ok():
            """确定按钮回调（校验输入合法性）"""
            # 1. 校验输入源（必须选择本地视频或摄像头）
            if not self.input_type:
                messagebox.showwarning("警告", "请选择输入源（本地视频或摄像头）")
                return
            # 2. 校验关节选择（至少选中一个）
            if not any(selected_joints_dict.values()):
                messagebox.showwarning("警告", "请选择至少一个关节")
                return
            # 3. 关闭对话框（继续执行分析流程）
            dialog.destroy()

        def on_cancel():
            """取消按钮回调（退出程序）"""
            self.input_source = None
            dialog.destroy()

        tk.Button(button_frame, text="确定", command=on_ok, width=15, bg="#4CAF50", fg="white").pack(side=tk.RIGHT, padx=10)
        tk.Button(button_frame, text="取消", command=on_cancel, width=15, bg="#f44336", fg="white").pack(side=tk.RIGHT, padx=10)

        # ------------------------------
        # 处理窗口关闭事件（点击右上角×）
        # ------------------------------
        def on_closing():
            self.input_source = None
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_closing)
        # 等待对话框关闭（模态，阻塞直到关闭）
        dialog.wait_window()

        # 返回是否选择了有效输入源（非None）
        return self.input_source is not None

    def read_video(self) -> Iterator[FrameData]:
        try:
            # 打开视频源
            self.cap = cv2.VideoCapture(self.input_source)
            if not self.cap.isOpened():
                raise ValueError(f"无法打开输入源：{self.input_source}")
            
            # 确定合适的帧率
            cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
            fps = cap_fps if cap_fps > 0 else app_config.video.default_fps
            
            frame_idx = 0
            self.processing = True
            
            while self.processing and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 精确的时间戳获取
                if self.input_type == "file":
                    timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                else:
                    timestamp = time.perf_counter()
                
                yield FrameData(frame=frame, timestamp=timestamp, fps=fps, frame_idx=frame_idx)
                frame_idx += 1
                
        finally:
            self.stop_processing()

    def stop_processing(self):
        """停止处理过程并释放资源"""
        self.processing = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def start(self):
        """启动运动分析流程（无需再检查输入源，因为main.py已校验）"""
        try:
            processor = MediaPipeProcessor()
            presentation = Presentation()
            frame_data_list = []
            joint_data_list = []

            # 创建预览窗口（可调整大小）
            cv2.namedWindow("运动分析 - 实时预览", cv2.WINDOW_NORMAL)

            # 处理每一帧（从read_video生成器获取）
            for frame_data in self.read_video():
                joint_data = processor.process_frame(frame_data)
                frame_data_list.append(frame_data)
                joint_data_list.append(joint_data)

                # 绘制关节线条与角度（依赖Presentation类）
                display_frame = frame_data.frame.copy()
                frame_height, frame_width = display_frame.shape[:2]
                presentation._draw_joint_lines(display_frame, joint_data, frame_width, frame_height)
                presentation._draw_joint_angles(display_frame, joint_data)

                # 显示预览帧
                cv2.imshow("运动分析 - 实时预览", display_frame)

                # 检查退出按键（q或ESC）
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    print("用户请求退出...")
                    break

            # 生成输出（CSV、可视化视频等）
            if frame_data_list and joint_data_list:
                presentation.generate_output(frame_data_list, joint_data_list, self.selected_joints)
                print("输出已生成（CSV、可视化视频）")
            else:
                print("未收集到有效数据（无帧或关节数据）")

        except Exception as e:
            messagebox.showerror("错误", f"处理失败：{str(e)}")
            import traceback
            traceback.print_exc()  # 打印详细错误栈（便于调试）
        finally:
            # 释放资源（关闭窗口、释放摄像头/视频文件）
            cv2.destroyAllWindows()
            self.stop_processing()