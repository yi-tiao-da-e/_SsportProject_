# video_io/input_handler.py
from typing import Iterator, Set
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
        self.selected_joints: Set[str] = set()  # 选择的关节部位

    def show_input_dialog(self) -> bool:
        try:
            root = tk.Tk()
            root.title("选择输入源和识别部位")
            root.geometry("800x600")
            root.resizable(False, False)
            root.attributes("-topmost", True)

            # 部位选择变量
            joint_vars = {}
            # 人型模型按钮字典
            human_buttons = {}
            
            def sync_joint_selection(joint_key, var):
                """同步左侧复选框和右侧人型模型按钮的选择状态"""
                def sync():
                    if joint_key in human_buttons:
                        button = human_buttons[joint_key]
                        if var.get():
                            button.config(bg='blue', relief=tk.SUNKEN)
                        else:
                            button.config(bg='red', relief=tk.RAISED)
                return sync
            
            def on_human_button_click(joint_key, var):
                """人型模型按钮点击事件"""
                def click():
                    new_state = not var.get()
                    var.set(new_state)
                    if joint_key in human_buttons:
                        button = human_buttons[joint_key]
                        if new_state:
                            button.config(bg='blue', relief=tk.SUNKEN)
                        else:
                            button.config(bg='red', relief=tk.RAISED)
                return click

            def select_file():
                self.input_source = filedialog.askopenfilename(
                    title="选择视频文件",
                    filetypes=[("视频文件", "*.mp4 *.avi *.mov")],
                    initialdir="./"
                )
                if self.input_source:
                    self.input_type = "file"
                    self._collect_selected_joints(joint_vars)
                    root.destroy()
                else:
                    messagebox.showwarning("警告", "未选择文件")

            def select_camera():
                self.input_source = 0
                self.input_type = "camera"
                self._collect_selected_joints(joint_vars)
                root.destroy()
            
            def on_closing():
                self.input_source = None
                root.destroy()

            def select_all_joints():
                for joint_key, var in joint_vars.items():
                    var.set(True)
                    if joint_key in human_buttons:
                        human_buttons[joint_key].config(bg='blue', relief=tk.SUNKEN)

            def clear_all_joints():
                for joint_key, var in joint_vars.items():
                    var.set(False)
                    if joint_key in human_buttons:
                        human_buttons[joint_key].config(bg='red', relief=tk.RAISED)

            root.protocol("WM_DELETE_WINDOW", on_closing)
            
            # 创建主框架
            main_frame = tk.Frame(root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
            
            # 输入源选择部分
            tk.Label(main_frame, text="请选择输入源：", font=("Arial", 12, "bold")).pack(pady=10)
            
            button_frame = tk.Frame(main_frame)
            button_frame.pack(pady=10)
            
            tk.Button(button_frame, text="本地视频文件", command=select_file, width=15, height=2, 
                     font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
            tk.Button(button_frame, text="摄像头", command=select_camera, width=15, height=2,
                     font=("Arial", 10)).pack(side=tk.LEFT, padx=10)

            # 分隔线
            separator = tk.Frame(main_frame, height=2, bd=1, relief=tk.SUNKEN)
            separator.pack(fill=tk.X, padx=10, pady=20)

            # 部位选择部分
            tk.Label(main_frame, text="选择要识别的部位：", font=("Arial", 12, "bold")).pack(pady=10)

            # 创建左右分栏框架
            selection_frame = tk.Frame(main_frame)
            selection_frame.pack(fill=tk.BOTH, expand=True, pady=10)

            # 左侧：复选框区域
            left_frame = tk.Frame(selection_frame, width=350)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))

            # 创建滚动框架用于部位选择
            joint_frame = tk.Frame(left_frame)
            joint_frame.pack(fill=tk.BOTH, expand=True)

            # 创建滚动条
            scrollbar = tk.Scrollbar(joint_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # 创建画布用于滚动
            canvas = tk.Canvas(joint_frame, yscrollcommand=scrollbar.set)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=canvas.yview)

            # 在画布上创建框架
            scrollable_frame = tk.Frame(canvas)
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

            # 右侧：火柴人模型区域
            right_frame = tk.Frame(selection_frame, width=400, height=400, bg='white', 
                                 relief=tk.RAISED, bd=2)
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            right_frame.pack_propagate(False)

            # 在火柴人模型区域创建画布
            human_canvas = tk.Canvas(right_frame, width=400, height=400, bg='white', 
                                   highlightthickness=1, highlightbackground="black")
            human_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # 立即绘制火柴人模型
            self._draw_stick_figure(human_canvas, human_buttons, joint_vars, on_human_button_click)

            # 定义所有可识别的关节部位
            joint_categories = {
                "上半身": [
                    ("左肩", "left_shoulder"),
                    ("左肘", "left_elbow"), 
                    ("左腕", "left_wrist"),
                    ("右肩", "right_shoulder"),
                    ("右肘", "right_elbow"),
                    ("右腕", "right_wrist")
                ],
                "躯干": [
                    ("颈椎", "neck"),
                    ("髋部中心", "hip_center")
                ],
                "下半身": [
                    ("左髋", "left_hip"),
                    ("左膝", "left_knee"),
                    ("左踝", "left_ankle"),
                    ("右髋", "right_hip"), 
                    ("右膝", "right_knee"),
                    ("右踝", "right_ankle")
                ]
            }

            # 创建复选框
            row = 0
            for category, joints in joint_categories.items():
                # 类别标签
                tk.Label(scrollable_frame, text=category, font=("Arial", 11, "bold"),
                        bg='lightcyan', relief=tk.RAISED, bd=1).grid(
                    row=row, column=0, columnspan=2, sticky="we", pady=(10, 5), padx=5
                )
                row += 1
                
                # 该类别下的关节
                for joint_name, joint_key in joints:
                    var = tk.BooleanVar(value=True)
                    joint_vars[joint_key] = var
                    
                    # 创建复选框
                    cb = tk.Checkbutton(scrollable_frame, text=joint_name, variable=var, 
                                      font=("Arial", 10),
                                      command=sync_joint_selection(joint_key, var),
                                      bg='white', activebackground='lightblue')
                    cb.grid(row=row, column=0, sticky="w", padx=20, pady=3)
                    row += 1

            # 全选/清除按钮
            button_frame2 = tk.Frame(scrollable_frame)
            button_frame2.grid(row=row, column=0, columnspan=2, pady=15)
            
            tk.Button(button_frame2, text="全选所有部位", command=select_all_joints, 
                     width=12, height=1, font=("Arial", 9), bg='lightgreen').pack(side=tk.LEFT, padx=8)
            tk.Button(button_frame2, text="清除所有部位", command=clear_all_joints, 
                     width=12, height=1, font=("Arial", 9), bg='lightcoral').pack(side=tk.LEFT, padx=8)

            # 更新滚动区域
            scrollable_frame.update_idletasks()
            canvas.config(scrollregion=canvas.bbox("all"))

            # 初始同步火柴人模型按钮状态
            for joint_key, var in joint_vars.items():
                if joint_key in human_buttons:
                    if var.get():
                        human_buttons[joint_key].config(bg='blue', relief=tk.SUNKEN)
                    else:
                        human_buttons[joint_key].config(bg='red', relief=tk.RAISED)

            # 添加提示文本
            tip_label = tk.Label(main_frame, text="提示：点击左侧复选框或右侧火柴人模型上的按钮来选择/取消选择部位", 
                               font=("Arial", 9), fg='gray')
            tip_label.pack(pady=10)

            # 添加图例说明
            legend_frame = tk.Frame(main_frame)
            legend_frame.pack(pady=5)
            
            # 蓝色按钮图例（选中）
            legend_button1 = tk.Button(legend_frame, width=4, height=1, bg='blue', relief=tk.SUNKEN)
            legend_button1.pack(side=tk.LEFT, padx=5)
            tk.Label(legend_frame, text="= 选中部位", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
            
            # 红色按钮图例（未选中）
            legend_button2 = tk.Button(legend_frame, width=4, height=1, bg='red', relief=tk.RAISED)
            legend_button2.pack(side=tk.LEFT, padx=(15, 5))
            tk.Label(legend_frame, text="= 未选部位", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)

            root.mainloop()
            return self.input_source is not None and len(self.selected_joints) > 0
        except Exception as e:
            print(f"输入对话框创建失败：{str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _draw_stick_figure(self, canvas, human_buttons, joint_vars, on_human_button_click):
        """在画布上绘制火柴人模型和可点击的按钮"""
        try:
            canvas_width = 400
            canvas_height = 400
            
            # 清除画布
            canvas.delete("all")
            
            # 设置背景色
            canvas.config(bg='white')
            
            # 火柴人各部位坐标
            center_x = canvas_width // 2
            joints_coords = {
                # 头部和躯干
                "neck": (center_x, 100),
                
                # 上半身 - 左侧
                "left_shoulder": (center_x - 50, 120),
                "left_elbow": (center_x - 80, 160),
                "left_wrist": (center_x - 60, 200),
                
                # 上半身 - 右侧
                "right_shoulder": (center_x + 50, 120),
                "right_elbow": (center_x + 80, 160),
                "right_wrist": (center_x + 60, 200),
                
                # 躯干中心
                "hip_center": (center_x, 200),
                
                # 下半身 - 左侧
                "left_hip": (center_x - 25, 200),
                "left_knee": (center_x - 30, 260),
                "left_ankle": (center_x - 30, 320),
                
                # 下半身 - 右侧
                "right_hip": (center_x + 25, 200),
                "right_knee": (center_x + 30, 260),
                "right_ankle": (center_x + 30, 320)
            }
            
            # 绘制火柴人骨架
            # 头部（圆形）
            canvas.create_oval(center_x - 20, 60, center_x + 20, 100, outline="black", width=2)
            
            # 躯干（直线）
            canvas.create_line(center_x, 100, center_x, 200, width=3, fill="black")
            
            # 手臂 - 左侧
            canvas.create_line(center_x, 120, center_x - 50, 120, width=2, fill="black")  # 肩线
            canvas.create_line(center_x - 50, 120, center_x - 80, 160, width=2, fill="black")  # 上臂
            canvas.create_line(center_x - 80, 160, center_x - 60, 200, width=2, fill="black")  # 前臂
            
            # 手臂 - 右侧
            canvas.create_line(center_x, 120, center_x + 50, 120, width=2, fill="black")  # 肩线
            canvas.create_line(center_x + 50, 120, center_x + 80, 160, width=2, fill="black")  # 上臂
            canvas.create_line(center_x + 80, 160, center_x + 60, 200, width=2, fill="black")  # 前臂
            
            # 腿部 - 左侧
            canvas.create_line(center_x, 200, center_x - 25, 200, width=2, fill="black")  # 髋线
            canvas.create_line(center_x - 25, 200, center_x - 30, 260, width=2, fill="black")  # 大腿
            canvas.create_line(center_x - 30, 260, center_x - 30, 320, width=2, fill="black")  # 小腿
            
            # 腿部 - 右侧
            canvas.create_line(center_x, 200, center_x + 25, 200, width=2, fill="black")  # 髋线
            canvas.create_line(center_x + 25, 200, center_x + 30, 260, width=2, fill="black")  # 大腿
            canvas.create_line(center_x + 30, 260, center_x + 30, 320, width=2, fill="black")  # 小腿
            
            # 在关节位置创建可交互按钮
            button_size = 80  # 按钮尺寸
            
            for joint_key, (x, y) in joints_coords.items():
                if joint_key in joint_vars:
                    var = joint_vars[joint_key]
                    
                    # 创建按钮 - 选中为蓝色，未选中为红色
                    button = tk.Button(canvas, 
                                     width=2, 
                                     height=1,
                                     bg='blue' if var.get() else 'red',  # 选中蓝色，未选中红色
                                     relief=tk.SUNKEN if var.get() else tk.RAISED,
                                     command=on_human_button_click(joint_key, var))
                    
                    # 将按钮放置在画布上
                    canvas.create_window(x, y, window=button, width=button_size*2, height=button_size*2)
                    human_buttons[joint_key] = button
                    
            print("火柴人模型绘制完成")  # 调试信息
                    
        except Exception as e:
            print(f"绘制火柴人模型时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _collect_selected_joints(self, joint_vars: dict):
        """收集用户选择的关节部位"""
        self.selected_joints.clear()
        for joint_key, var in joint_vars.items():
            if var.get():
                self.selected_joints.add(joint_key)

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
            messagebox.showerror("错误", "未选择输入源或识别部位，应用退出")
            return

        try:
            # 传递选择的部位到处理器
            processor = MediaPipeProcessor(selected_joints=self.selected_joints)
            presentation = Presentation(selected_joints=self.selected_joints)
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
                
                # 绘制关节识别结果（只绘制选中的部位）
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
            # 关闭opencv窗口
            cv2.destroyAllWindows()
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
            self.stop_processing()