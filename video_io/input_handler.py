# video_io/input_handler.py
from typing import Iterator
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import time
from dataclasses import dataclass
from config.settings import app_config
# 1. 仅导入MediaPipeProcessor，不导入JointData
from recog.med_processor import MediaPipeProcessor
# 2. 从独立文件导入FrameData（关键！）
from video_io.data_classes import FrameData  # 这里是循环导入的根源！必须改！
from video_io.presentation import Presentation

@dataclass
class FrameData:
    frame: cv2.Mat
    timestamp: float
    fps: float
    frame_idx: int

class InputHandler:
    def __init__(self):
        self.input_source = None
        self.input_type = None

    def show_input_dialog(self) -> bool:
        try:
            root = tk.Tk()
            root.title("选择输入源")
            root.geometry("350x180")
            root.resizable(False, False)

            def select_file():
                self.input_source = filedialog.askopenfilename(
                    title="选择视频文件",
                    filetypes=[("视频文件", "*.mp4 *.avi *.mov")],
                    initialdir="./"
                )
                if self.input_source:
                    self.input_type = "file"
                    root.destroy()
                else:
                    messagebox.showwarning("警告", "未选择文件")

            def select_camera():
                self.input_source = 0
                self.input_type = "camera"
                root.destroy()

            tk.Label(root, text="请选择输入源：", font=("Arial", 12)).pack(pady=10)
            tk.Button(root, text="本地视频文件", command=select_file, width=20, height=2).pack(pady=5)
            tk.Button(root, text="摄像头", command=select_camera, width=20, height=2).pack(pady=5)
            root.mainloop()
            return self.input_source is not None
        except Exception as e:
            print(f"输入对话框创建失败：{str(e)}")
            return False

    def read_video(self) -> Iterator[FrameData]:
        cap = cv2.VideoCapture(self.input_source)
        if not cap.isOpened():
            raise ValueError(f"无法打开输入源：{self.input_source}")

        fps = cap.get(cv2.CAP_PROP_FPS) if self.input_type == "file" else app_config.video.default_fps
        fps = fps or app_config.video.default_fps
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000 if self.input_type == "file" else time.time()
            yield FrameData(frame=frame, timestamp=timestamp, fps=fps, frame_idx=frame_idx)
            frame_idx += 1
        cap.release()

    # video_io/input_handler.py
    def start(self):
        if not self.show_input_dialog():
            messagebox.showerror("错误", "未选择输入源，应用退出")
            return

        try:
            processor = MediaPipeProcessor()
            frame_data_list = []
            joint_data_list = []

            # 读取并处理每帧（强制每帧添加joint_data）
            for frame_data in self.read_video():
                frame_data_list.append(frame_data)
                joint_data = processor.process_frame(frame_data)
                joint_data_list.append(joint_data)  # 无需判断，直接添加

            # 预览窗口（按q退出）
                cv2.imshow("输入预览", frame_data.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 验证数据量（必一致）
            print(f"收集到{len(frame_data_list)}帧原始数据，{len(joint_data_list)}帧关节数据")
        
            # 生成输出（无需再判断数量是否一致）
            if frame_data_list and joint_data_list:
                presentation = Presentation()
                presentation.generate_output(frame_data_list, joint_data_list)
                print("输出已生成")
            else:
                print("未收集到有效数据")

        except Exception as e:
            messagebox.showerror("错误", f"处理失败：{str(e)}")
        finally:
            cv2.destroyAllWindows()