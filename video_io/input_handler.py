# video_io/input_handler.py
from typing import Iterator
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import time
from dataclasses import dataclass
from config.settings import app_config
from recog.med_processor import MediaPipeProcessor, JointData  # 假设med_processor.py正确
from video_io.presentation import Presentation  # 假设presentation.py正确

@dataclass
class FrameData:
    """结构化帧数据（传递给后续模块）"""
    frame: cv2.Mat  # BGR格式原始帧
    timestamp: float  # 帧捕获时间（秒）
    fps: float  # 输入源帧率
    frame_idx: int  # 帧索引（从0开始）

class InputHandler:
    def __init__(self):
        self.input_source = None  # 输入源（文件路径或摄像头索引）
        self.input_type = None    # 输入类型（"file"或"camera"）

        # video_io/input_handler.py
    def show_input_dialog(self) -> bool:
        """显示用户交互窗口，选择输入源（本地文件/摄像头）"""
        try:
            root = tk.Tk()
            root.title("选择输入源")
            root.geometry("300x150")  # 窗口大小（宽x高）
        
        # 定义选择文件的回调函数
            def select_file():
                self.input_source = filedialog.askopenfilename(
                    title="选择视频文件",
                    filetypes=[("视频文件", "*.mp4 *.avi *.mov")],
                    initialdir="./"  # 设置默认打开目录（可选，避免路径错误）
                )
                if self.input_source:
                    self.input_type = "file"
                    root.destroy()  # 关闭窗口
                else:
                    messagebox.showwarning("警告", "未选择文件")
        
        # 定义选择摄像头的回调函数
            def select_camera():
                self.input_source = 0  # 默认摄像头索引（0为第一个摄像头）
                self.input_type = "camera"
                root.destroy()  # 关闭窗口
        
            # 添加按钮（调整布局，确保可见）
            tk.Button(root, text="本地视频文件", command=select_file).pack(pady=20)  # 增加上下边距（pady=20）
            tk.Button(root, text="摄像头", command=select_camera).pack(pady=10)
        
            # 启动tkinter事件循环（必须调用，否则窗口无法显示）
            root.mainloop()
        
            return self.input_source is not None  # 返回是否成功选择
        
        except Exception as e:
            print(f"输入对话框创建失败：{str(e)}")  # 打印异常信息
            return False

    def read_video(self) -> Iterator[FrameData]:
        """读取视频帧,生成FrameData(生成器)"""
        cap = cv2.VideoCapture(self.input_source)
        if not cap.isOpened():
            raise ValueError(f"无法打开输入源：{self.input_source}")
        
        # 获取帧率（本地文件用视频元数据，摄像头用默认配置）
        if self.input_type == "file":
            fps = cap.get(cv2.CAP_PROP_FPS) or app_config.video.default_fps
        else:
            fps = app_config.video.default_fps
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 获取时间戳（本地文件用视频位置，摄像头用系统时间）
            if self.input_type == "file":
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 转换为秒
            else:
                timestamp = time.time()  # 系统时间（秒）
            
            # 生成FrameData并yield
            yield FrameData(
                frame=frame,
                timestamp=timestamp,
                fps=fps,
                frame_idx=frame_idx
            )
            frame_idx += 1
        
        cap.release()

    def start(self):
        """启动输入处理流程（用户交互→视频读取→关节处理→输出）"""
        if not self.show_input_dialog():
            messagebox.showerror("错误", "未选择输入源，应用退出")
            return
        
        try:
            # 1. 初始化处理器与数据列表
            processor = MediaPipeProcessor()
            frame_data_list = []
            joint_data_list = []

            # 2. 读取并处理每帧
            for frame_data in self.read_video():
                # 2.1 添加原始帧数据到列表
                frame_data_list.append(frame_data)
                
                # 2.2 处理帧数据，获取关节数据
                joint_data = processor.process_frame(frame_data)
                if joint_data:
                    joint_data_list.append(joint_data)
                
                # 2.3 显示输入预览（可选，按q退出）
                cv2.imshow("输入预览", frame_data.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 3. 打印数据量（排查用）
            print(f"收集到{len(frame_data_list)}帧原始数据，{len(joint_data_list)}帧关节数据")
            
            # 4. 验证数据一致性
            if len(frame_data_list) != len(joint_data_list):
                print(f"数据不一致：原始帧{len(frame_data_list)}帧，关节数据{len(joint_data_list)}帧")
                return
            
            # 5. 生成输出（视频+CSV）
            if frame_data_list and joint_data_list:
                presentation = Presentation()
                presentation.generate_output(frame_data_list, joint_data_list)
                print("输出已生成")
            else:
                print("未收集到有效数据，无法生成输出")

        except Exception as e:
            messagebox.showerror("错误", f"处理失败：{str(e)}")
        finally:
            cv2.destroyAllWindows()