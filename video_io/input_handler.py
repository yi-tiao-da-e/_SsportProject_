# video_io/input_handler.py
from typing import Iterator
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import time
from dataclasses import dataclass
from config.settings import app_config
from recog.med_processor import MediaPipeProcessor
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
        self.processing = False  # 处理状态标志
        self.cap = None  # 视频捕获对象

    def show_input_dialog(self) -> bool:
        try:
            root = tk.Tk()
            root.title("选择输入源")
            root.geometry("350x180")
            root.resizable(False, False)
            root.attributes("-topmost", True)  # 确保对话框在最前

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
            
            def on_closing():
                self.input_source = None
                root.destroy()

            root.protocol("WM_DELETE_WINDOW", on_closing)
            
            tk.Label(root, text="请选择输入源：", font=("Arial", 12)).pack(pady=10)
            tk.Button(root, text="本地视频文件", command=select_file, width=20, height=2).pack(pady=5)
            tk.Button(root, text="摄像头", command=select_camera, width=20, height=2).pack(pady=5)
            root.mainloop()
            return self.input_source is not None
        except Exception as e:
            print(f"输入对话框创建失败：{str(e)}")
            return False

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
        if not self.show_input_dialog():
            messagebox.showerror("错误", "未选择输入源，应用退出")
            return

        try:
            processor = MediaPipeProcessor()
            presentation = Presentation()
            frame_data_list = []
            joint_data_list = []
            
            # 创建预览窗口
            cv2.namedWindow("运动分析 - 实时预览", cv2.WINDOW_NORMAL)
            
            # 处理每一帧
            for frame_data in self.read_video():
                # 处理当前帧
                joint_data = processor.process_frame(frame_data)
                
                # 保存数据用于后续输出
                frame_data_list.append(frame_data)
                joint_data_list.append(joint_data)
                
                # 创建带关节显示的帧
                display_frame = frame_data.frame.copy()
                
                # 绘制关节识别结果
                frame_height, frame_width = display_frame.shape[:2]
                presentation._draw_joint_lines(display_frame, joint_data, frame_width, frame_height)
                presentation._draw_joint_angles(display_frame, joint_data)
                
                # 显示处理后的帧
                cv2.imshow("运动分析 - 实时预览", display_frame)
                
                # 检查退出按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 27是ESC键
                    print("用户请求退出...")
                    break
                    
            # 验证数据量
            print(f"收集到{len(frame_data_list)}帧原始数据，{len(joint_data_list)}帧关节数据")
        
            # 生成输出文件
            if frame_data_list and joint_data_list:
                presentation.generate_output(frame_data_list, joint_data_list)
                print("输出已生成")
            else:
                print("未收集到有效数据")

        except Exception as e:
            messagebox.showerror("错误", f"处理失败：{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            self.stop_processing()