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
        self.circle_items = {}  # 存储圆形对象的ID

    def show_input_dialog(self) -> bool:
        try:
            root = tk.Tk()
            root.title("选择输入源和识别部位")
            root.geometry("850x650")
            root.resizable(False, False)
            root.attributes("-topmost", True)

            # 部位选择变量
            joint_vars = {}
            
            # 定义所有可识别的关节部位
            self.joint_categories = {
                "上半身": [
                    ("左肩", "left_shoulder"),
                    ("左肘", "left_elbow"), 
                    ("左腕", "left_wrist"),
                    ("右肩", "right_shoulder"),
                    ("右肘", "right_elbow"),
                    ("右腕", "right_wrist"),
                    ("颈椎", "neck")
                ],
                "脊柱": [
                    ("颈椎", "neck"),
                    ("胸椎", "mid_spine"),
                    ("腰椎", "low_spine")
                ],
                "下半身": [
                    ("髋部中心", "hip_center"),
                    ("左髋", "left_hip"),
                    ("左膝", "left_knee"),
                    ("左踝", "left_ankle"),
                    ("右髋", "right_hip"), 
                    ("右膝", "right_knee"),
                    ("右踝", "right_ankle")
                ]
            }
            
            # 初始化关节变量
            for category, joints in self.joint_categories.items():
                for joint_name, joint_key in joints:
                    # 防止重复初始化同名关节
                    if joint_key not in joint_vars:
                        joint_vars[joint_key] = tk.BooleanVar(value=True)
            
            def sync_joint_selection(joint_key, var):
                """同步左侧复选框和右侧人型模型按钮的选择状态"""
                def sync():
                    if joint_key in self.circle_items:
                        canvas = human_canvas
                        if var.get():
                            canvas.itemconfig(self.circle_items[joint_key], fill='blue')
                        else:
                            canvas.itemconfig(self.circle_items[joint_key], fill='red')
                return sync
            
            def on_human_button_click(joint_key, var):
                """人型模型按钮点击事件"""
                def click(event=None):
                    new_state = not var.get()
                    var.set(new_state)
                    if joint_key in self.circle_items:
                        canvas = human_canvas
                        if new_state:
                            canvas.itemconfig(self.circle_items[joint_key], fill='blue')
                        else:
                            canvas.itemconfig(self.circle_items[joint_key], fill='red')
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

            def toggle_category_selection(category):
                """切换类别的全选/取消全选状态"""
                joints = self.joint_categories[category]
                # 检查当前是否已全选
                all_selected = all(joint_vars[joint_key].get() for _, joint_key in joints)
                
                # 设置新状态
                new_state = not all_selected
                for _, joint_key in joints:
                    var = joint_vars[joint_key]
                    var.set(new_state)
                    if joint_key in self.circle_items:
                        canvas = human_canvas
                        if new_state:
                            canvas.itemconfig(self.circle_items[joint_key], fill='blue')
                        else:
                            canvas.itemconfig(self.circle_items[joint_key], fill='red')
            
            def toggle_all_selection():
                """切换所有关节的全选/取消全选状态"""
                # 检查当前是否已全选
                all_selected = all(var.get() for var in joint_vars.values())
                
                # 设置新状态
                new_state = not all_selected
                for joint_key, var in joint_vars.items():
                    var.set(new_state)
                    if joint_key in self.circle_items:
                        canvas = human_canvas
                        if new_state:
                            canvas.itemconfig(self.circle_items[joint_key], fill='blue')
                        else:
                            canvas.itemconfig(self.circle_items[joint_key], fill='red')

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
            self._draw_stick_figure(human_canvas, joint_vars, on_human_button_click)
            
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

            # 创建复选框
            row = 0
            for category, joints in self.joint_categories.items():
                # 类别标签框架
                category_frame = tk.Frame(scrollable_frame)
                category_frame.grid(row=row, column=0, columnspan=2, sticky="we", pady=(10, 5))
                row += 1
                
                # 类别标签
                tk.Label(category_frame, text=category, font=("Arial", 11, "bold"),
                        bg='lightcyan', relief=tk.RAISED, bd=1).pack(side=tk.LEFT, padx=5)
                
                # 类别全选按钮（放在右侧）
                toggle_btn = tk.Button(category_frame, text="全选/取消", 
                                     command=lambda c=category: toggle_category_selection(c),
                                     width=8, height=1, font=("Arial", 9), bg='lightblue')
                toggle_btn.pack(side=tk.RIGHT, padx=5)
                
                # 该类别下的关节
                for joint_name, joint_key in joints:
                    # 确保关节键存在
                    if joint_key in joint_vars:
                        var = joint_vars[joint_key]
                        
                        # 创建复选框
                        cb = tk.Checkbutton(scrollable_frame, text=joint_name, variable=var, 
                                          font=("Arial", 10),
                                          command=sync_joint_selection(joint_key, var),
                                          bg='white', activebackground='lightblue')
                        cb.grid(row=row, column=0, sticky="w", padx=20, pady=3)
                        row += 1

            # 操作按钮框架 - 只保留全局切换按钮
            button_frame2 = tk.Frame(scrollable_frame)
            button_frame2.grid(row=row, column=0, columnspan=2, pady=15)
            row += 1
            
            # 全局全选/取消按钮
            tk.Button(button_frame2, text="全选/取消", command=toggle_all_selection, 
                     width=10, height=1, font=("Arial", 9), bg='lightgreen').pack(pady=10)

            # 更新滚动区域
            scrollable_frame.update_idletasks()
            canvas.config(scrollregion=canvas.bbox("all"))

            # 初始同步火柴人模型按钮状态
            for joint_key, var in joint_vars.items():
                if joint_key in self.circle_items:
                    canvas = human_canvas
                    if var.get():
                        canvas.itemconfig(self.circle_items[joint_key], fill='blue')
                    else:
                        canvas.itemconfig(self.circle_items[joint_key], fill='red')

            # 添加提示文本
            tip_label = tk.Label(main_frame, text="提示：点击左侧复选框或右侧火柴人模型上的按钮来选择/取消选择部位", 
                               font=("Arial", 9), fg='gray')
            tip_label.pack(pady=10)

            # 添加图例说明
            legend_frame = tk.Frame(main_frame)
            legend_frame.pack(pady=5)
            
            # 蓝色按钮图例（选中）
            tk.Canvas(legend_frame, width=24, height=24, highlightthickness=0, bg='white').create_oval(2, 2, 22, 22, fill='blue', outline='black')
            tk.Label(legend_frame, text="= 选中部位", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
            
            # 红色按钮图例（未选中）
            tk.Canvas(legend_frame, width=24, height=24, highlightthickness=0, bg='white').create_oval(2, 2, 22, 22, fill='red', outline='black')
            tk.Label(legend_frame, text="= 未选部位", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)

            root.mainloop()
            return self.input_source is not None and len(self.selected_joints) > 0
        except Exception as e:
            print(f"输入对话框创建失败：{str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _draw_stick_figure(self, canvas, joint_vars, on_human_button_click):
        """在画布上绘制火柴人模型和可点击的圆形按钮"""
        try:
            self.circle_items = {}
            canvas_width = 400
            canvas_height = 400
            
            # 清除画布
            canvas.delete("all")
            
            # 设置背景色
            canvas.config(bg='white')
            
            # 火柴人各部位坐标
            center_x = canvas_width // 2
            center_y = canvas_height // 2
            
            # 更合理的火柴人关节坐标
            joints_coords = {
                # 头部和躯干
                "neck": (center_x, 80),
                "mid_spine": (center_x, 130),  # 胸椎位置
                "low_spine": (center_x, 180),  # 腰椎位置
                
                # 上半身 - 左侧
                "left_shoulder": (center_x - 40, 100),
                "left_elbow": (center_x - 70, 140),
                "left_wrist": (center_x - 50, 180),
                
                # 上半身 - 右侧
                "right_shoulder": (center_x + 40, 100),
                "right_elbow": (center_x + 70, 140),
                "right_wrist": (center_x + 50, 180),
                
                # 躯干中心和髋部
                "hip_center": (center_x, 200),
                "left_hip": (center_x - 20, 200),
                "right_hip": (center_x + 20, 200),
                
                # 下半身 - 左侧
                "left_knee": (center_x - 25, 260),
                "left_ankle": (center_x - 25, 320),
                
                # 下半身 - 右侧
                "right_knee": (center_x + 25, 260),
                "right_ankle": (center_x + 25, 320)
            }
            
            # 绘制火柴人骨架
            # 头部（圆形）
            canvas.create_oval(center_x - 20, 40, center_x + 20, 80, outline="black", width=2)
            
            # 脊柱（颈椎到胸椎到腰椎）
            canvas.create_line(center_x, 80, center_x, 130, width=3, fill="black")  # 颈椎到胸椎
            canvas.create_line(center_x, 130, center_x, 180, width=3, fill="black")  # 胸椎到腰椎
            canvas.create_line(center_x, 180, center_x, 200, width=3, fill="black")  # 腰椎到髋部中心
            
            # 手臂 - 左侧
            canvas.create_line(center_x, 100, center_x - 40, 100, width=2, fill="black")  # 肩线
            canvas.create_line(center_x - 40, 100, center_x - 70, 140, width=2, fill="black")  # 上臂
            canvas.create_line(center_x - 70, 140, center_x - 50, 180, width=2, fill="black")  # 前臂
            
            # 手臂 - 右侧
            canvas.create_line(center_x, 100, center_x + 40, 100, width=2, fill="black")  # 肩线
            canvas.create_line(center_x + 40, 100, center_x + 70, 140, width=2, fill="black")  # 上臂
            canvas.create_line(center_x + 70, 140, center_x + 50, 180, width=2, fill="black")  # 前臂
            
            # 腿部 - 左侧
            canvas.create_line(center_x - 20, 200, center_x - 25, 260, width=2, fill="black")  # 大腿
            canvas.create_line(center_x - 25, 260, center_x - 25, 320, width=2, fill="black")  # 小腿
            
            # 腿部 - 右侧
            canvas.create_line(center_x + 20, 200, center_x + 25, 260, width=2, fill="black")  # 大腿
            canvas.create_line(center_x + 25, 260, center_x + 25, 320, width=2, fill="black")  # 小腿
            
            # 在关节位置创建可交互按钮
            button_radius = 12  # 按钮半径
            
            for joint_key, (x, y) in joints_coords.items():
                if joint_key in joint_vars:
                    var = joint_vars[joint_key]
                    
                    # 直接绘制圆形（按钮）
                    fill_color = 'blue' if var.get() else 'red'
                    
                    # 创建圆形项
                    circle_id = canvas.create_oval(
                        x - button_radius, y - button_radius,
                        x + button_radius, y + button_radius,
                        fill=fill_color,
                        outline='black',
                        width=1
                    )
                    
                    # 存储圆形ID以便后续更新
                    self.circle_items[joint_key] = circle_id
                    
                    # 绑定点击事件
                    canvas.tag_bind(circle_id, "<Button-1>", on_human_button_click(joint_key, var))
    
            # 强制画布更新
            canvas.update_idletasks()
            print(f"成功创建 {len(self.circle_items)} 个圆形关节按钮")
                    
        except Exception as e:
            print(f"绘制火柴人模型时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 尝试恢复显示文本信息
            canvas.create_text(200, 200, text="火柴人模型创建失败", font=("Arial", 16), fill="red")
            canvas.create_text(200, 230, text=f"错误: {str(e)}", font=("Arial", 10), fill="red")
            canvas.update_idletasks()

    def _collect_selected_joints(self, joint_vars: dict):
        """收集用户选择的关节部位"""
        self.selected_joints.clear()
        for joint_key, var in joint_vars.items():
            if var.get():
                self.selected_joints.add(joint_key)
        print(f"已选择关节: {', '.join(self.selected_joints)}")

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
            print(f"开始读取视频，FPS: {fps}")
            
            while self.processing and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("视频读取结束或发生错误")
                    break
                
                # 精确的时间戳获取
                if self.input_type == "file":
                    timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                else:
                    timestamp = time.perf_counter()
                
                yield FrameData(frame=frame, timestamp=timestamp, fps=fps, frame_idx=frame_idx)
                frame_idx += 1
                
                # 每100帧打印一次进度
                if frame_idx % 100 == 0:
                    print(f"已处理 {frame_idx} 帧")
                
        except Exception as e:
            print(f"视频读取错误: {str(e)}")
            raise
        finally:
            self.stop_processing()

    def stop_processing(self):
        """停止处理过程并释放资源"""
        print("停止处理并释放资源...")
        self.processing = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def start(self):
        if not self.show_input_dialog():
            messagebox.showerror("错误", "未选择输入源或识别部位，应用退出")
            return

        try:
            # 传递选择的部位到处理器
            print(f"初始化处理器，选择关节: {self.selected_joints}")
            processor = MediaPipeProcessor(selected_joints=self.selected_joints)
            presentation = Presentation(selected_joints=self.selected_joints)
            frame_data_list = []
            joint_data_list = []
            
            # 创建预览窗口
            cv2.namedWindow("运动分析 - 实时预览", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("运动分析 - 实时预览", 800, 600)
            
            print("开始处理视频...")
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
                
                # 显示帧信息
                cv2.putText(display_frame, f"Frame: {frame_data.frame_idx}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
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
                print("生成输出文件...")
                presentation.generate_output(frame_data_list, joint_data_list)
                print("输出已生成")
                messagebox.showinfo("完成", "分析完成，输出文件已生成！")
            else:
                print("未收集到有效数据")
                messagebox.showwarning("警告", "未收集到有效数据，无法生成输出")

        except Exception as e:
            print(f"处理失败: {str(e)}")
            messagebox.showerror("错误", f"处理失败：{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_processing()
            print("应用结束")