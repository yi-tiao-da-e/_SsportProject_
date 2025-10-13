import cv2
import pandas as pd
import os
from typing import List, Optional, Dict
from dataclasses import dataclass, field, asdict  # 新增：导入 asdict
from config.settings import app_config
from video_io.data_classes import FrameData
from recog.data_classes import JointData
import tkinter as tk
from tkinter import messagebox
@dataclass
class Presentation:
    """负责生成可视化视频、CSV表格及结果窗口的模块（`dataclass`风格）"""
    # ------------------------------
    # 从全局配置中获取输出路径（无需硬编码）
    # ------------------------------
    output_video_path: str = field(
        default_factory=lambda: app_config.video.output_video_path,
        metadata={"描述": "可视化视频输出路径"}
    )
    output_csv_path: str = field(
        default_factory=lambda: app_config.video.output_csv_path,
        metadata={"描述": "关节数据CSV输出路径"}
    )

    # ------------------------------
    # 可视化配置（在`__post_init__`中初始化）
    # ------------------------------
    joint_visual_config: dict = field(default_factory=dict)  # 关节绘制样式（颜色、厚度、关键点）
    text_config: dict = field(default_factory=dict)          # 文本样式（字体、颜色）

    def __post_init__(self):
        """`dataclass`的特殊方法，在`__init__`后自动执行（初始化配置）"""
        # 1. 确保输出目录存在（避免因目录不存在报错）
        os.makedirs(os.path.dirname(self.output_video_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)

        # 2. 初始化关节绘制样式（扩展为**所有主要关节**）
        self.joint_visual_config = {
            "shoulder": {  # 肩（颈→肩→肘）
                "color": (0, 255, 0),      # 绿色
                "thickness": 2,
                "points": ("neck", "shoulder", "elbow")  # 关键点顺序（近端→关节→远端）
            },
            "elbow": {     # 肘（肩→肘→腕）
                "color": (0, 0, 255),      # 红色（保留原颜色）
                "thickness": 2,
                "points": ("shoulder", "elbow", "wrist")
            },
            "wrist": {     # 腕（肘→腕→拇指）
                "color": (255, 0, 0),      # 蓝色
                "thickness": 2,
                "points": ("elbow", "wrist", "thumb")
            },
            "hip": {       # 髋（髋→膝→踝，暂用原逻辑）
                "color": (255, 255, 0),    # 黄色
                "thickness": 2,
                "points": ("hip", "knee", "ankle")
            },
            "knee": {      # 膝（髋→膝→踝，保留原颜色）
                "color": (255, 0, 255),    # 紫色
                "thickness": 2,
                "points": ("hip", "knee", "ankle")
            },
            "ankle": {     # 踝（膝→踝→脚跟）
                "color": (0, 255, 255),    # 青色
                "thickness": 2,
                "points": ("knee", "ankle", "heel")
            }
        }

        # 3. 初始化文本样式（修正单位，更符合用户习惯）
        self.text_config = {
            "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
            "fontScale": 0.5,
            "color": (255, 255, 255),  # 白色文本（对比强烈）
            "thickness": 1,
            "lineType": cv2.LINE_AA    # 抗锯齿（文本更清晰）
        }

    def _draw_joint_lines(self, frame: cv2.Mat, joint_data: JointData, frame_width: int, frame_height: int):
        """绘制所有关节连线（未选择的关节因数据为空自动跳过）"""
        # 遍历所有关节的可视化配置（如shoulder、elbow）
        for joint_name, config in self.joint_visual_config.items():
            # 处理左右侧（如left_shoulder、right_shoulder）
            for side in ["left", "right"]:
                points = []  # 存储关节关键点的像素坐标
                # 提取关节关键点的归一化坐标（如neck_x、left_shoulder_x）
                for point in config["points"]:
                    # 特殊处理无侧点（如neck，无需加left/right前缀）
                    if point == "neck":
                        x_field = f"{point}_x"
                        y_field = f"{point}_y"
                    else:
                        x_field = f"{side}_{point}_x"
                        y_field = f"{side}_{point}_y"
                    # 获取坐标（未选择的关节会返回None）
                    x = getattr(joint_data, x_field, None)
                    y = getattr(joint_data, y_field, None)
                    # 坐标无效（未检测到或未选择），跳过该关节
                    if x is None or y is None:
                        break
                    # 转换为像素坐标（归一化→帧宽高）
                    pixel_x = int(x * frame_width)
                    pixel_y = int(y * frame_height)
                    points.append((pixel_x, pixel_y))
                # 至少2个点才能绘制连线（如颈→肩→肘需要3个点）
                if len(points) >= 2:
                    # 绘制连续连线（如颈→肩→肘）
                    for i in range(len(points) - 1):
                        cv2.line(
                            frame, 
                            points[i], 
                            points[i+1], 
                            config["color"], 
                            config["thickness"]
                        )
    def _draw_joint_angles(self, frame: cv2.Mat, joint_data: JointData):
        """绘制所有关节的角度与角速度（未选择的关节因数据为空自动跳过）"""
        text_pos = (10, 30)  # 初始文本位置（左上角）
        line_spacing = 20     # 行间距（像素）
        # 定义关节显示顺序（避免重叠）
        joint_order = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
        
        # 遍历所有关节（如shoulder、elbow）
        for joint_name in joint_order:
            # 处理左右侧（如left_shoulder、right_shoulder）
            for side in ["left", "right"]:
                # 绘制角度（如left_shoulder_angle）
                angle_field = f"{side}_{joint_name}_angle"
                angle = getattr(joint_data, angle_field, None)
                if angle is not None:
                    text = f"{side.capitalize()} {joint_name.capitalize()} Angle: {angle:.1f}°"
                    cv2.putText(frame, text, text_pos, **self.text_config)
                    text_pos = (text_pos[0], text_pos[1] + line_spacing)  # 下移一行
                
                # 绘制角速度（如left_shoulder_velocity）
                velocity_field = f"{side}_{joint_name}_velocity"
                velocity = getattr(joint_data, velocity_field, None)
                if velocity is not None:
                    text = f"{side.capitalize()} {joint_name.capitalize()} Vel: {velocity:.1f}°/s"
                    cv2.putText(frame, text, text_pos, **self.text_config)
                    text_pos = (text_pos[0], text_pos[1] + line_spacing)  # 下移一行
    
    def generate_joint_csv(self, joint_data_list: List[JointData], selected_joints: List[Dict]):
        """生成CSV（仅包含选中关节数据）"""
        if not joint_data_list:
            raise ValueError("关节数据为空")
        if not selected_joints:
            raise ValueError("未选择任何关节")

        # 生成字段（基础字段+选中关节字段）
        base_fields = ["frame_idx", "timestamp", "fps"]
        joint_fields = []
        for joint in selected_joints:
            side = joint["side"]
            part = joint["part"]
            joint_fields.extend([
                f"{side}_{part}_x",
                f"{side}_{part}_y",
                f"{side}_{part}_angle",
                f"{side}_{part}_velocity"
            ])
        fields = base_fields + joint_fields

        # 生成DataFrame并保存
        df = pd.DataFrame([asdict(jd) for jd in joint_data_list])[fields]
        df.to_csv(self.output_csv_path, index=False)
        print(f"关节数据CSV已保存至：{self.output_csv_path}（仅包含选择的关节）")
    
    def calculate_motion_statistics(self, frame_data_list: List[FrameData], joint_data_list: List[JointData], selected_joints: List[Dict]):
        """计算选中关节的运动统计信息"""
        if not frame_data_list or not joint_data_list:
            return None
        if not selected_joints:
            return None

        # 计算运动时长
        start_time = frame_data_list[0].timestamp
        end_time = frame_data_list[-1].timestamp
        duration = end_time - start_time

        # 初始化最大角速度
        max_velocities = {}
        for joint in selected_joints:
            side = joint["side"]
            part = joint["part"]
            max_velocities[f"{side}_{part}_velocity"] = float('-inf')

        # 遍历帧计算最大角速度
        for joint_data in joint_data_list:
            for joint in selected_joints:
                side = joint["side"]
                part = joint["part"]
                velocity = getattr(joint_data, f"{side}_{part}_velocity", None)
                if velocity is not None and velocity > max_velocities[f"{side}_{part}_velocity"]:
                    max_velocities[f"{side}_{part}_velocity"] = velocity

        # 处理无有效数据的情况
        for key in max_velocities:
            if max_velocities[key] == float('-inf'):
                max_velocities[key] = 0.0

        # 返回统计结果
        return {
            "duration": duration,
            "max_velocities": max_velocities
        }
    
    def show_results_window(self, statistics: dict, selected_joints: List[str]):
        """显示运动分析结果窗口（仅包含用户选择的关节）"""
        if not statistics or not selected_joints:
            return
        
        try:
            # 创建结果窗口（调整大小以适应更多关节）
            root = tk.Tk()
            root.title("运动分析结果")
            root.geometry("500x400")
            root.resizable(False, False)
            
            # 设置窗口居中
            root.update_idletasks()
            x = (root.winfo_screenwidth() // 2) - (500 // 2)
            y = (root.winfo_screenheight() // 2) - (400 // 2)
            root.geometry(f"500x400+{x}+{y}")
            
            # 添加标题
            title_label = tk.Label(root, text="运动分析结果", font=("Arial", 16, "bold"))
            title_label.pack(pady=20)
            
            # 添加运动时长（必显示）
            duration_label = tk.Label(
                root, 
                text=f"运动时长: {statistics['duration']:.2f} 秒", 
                font=("Arial", 14)
            )
            duration_label.pack(pady=10)
            
            # 添加最大角速度标题
            velocity_title = tk.Label(
                root, 
                text="最大关节角速度", 
                font=("Arial", 12, "bold")
            )
            velocity_title.pack(pady=(20, 10))
            
            # 创建框架（网格布局，每行显示左右侧关节）
            velocity_frame = tk.Frame(root)
            velocity_frame.pack(pady=10)
            
            # 遍历选择的关节，显示左右侧的最大角速度
            row = 0
            for joint in selected_joints:
                # 左侧关节（如left_shoulder）
                left_key = f"left_{joint}"
                left_value = statistics["max_velocities"].get(left_key, 0.0)
                left_label = tk.Label(
                    velocity_frame,
                    text=f"左{joint}：{left_value:.1f} °/s",
                    font=("Arial", 11)
                )
                left_label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
                
                # 右侧关节（如right_shoulder）
                right_key = f"right_{joint}"
                right_value = statistics["max_velocities"].get(right_key, 0.0)
                right_label = tk.Label(
                    velocity_frame,
                    text=f"右{joint}：{right_value:.1f} °/s",
                    font=("Arial", 11)
                )
                right_label.grid(row=row, column=1, padx=10, pady=5, sticky="w")
                
                # 换行（处理下一个关节）
                row += 1
            
            # 添加确定按钮（关闭窗口）
            ok_button = tk.Button(root, text="确定", command=root.destroy, width=10, height=2)
            ok_button.pack(pady=20)
            
            # 启动窗口主循环（阻塞主线程，直到关闭）
            root.mainloop()
            
        except Exception as e:
            print(f"显示结果窗口时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def generate_output(self, frame_data_list: List[FrameData], joint_data_list: List[JointData], selected_joints: List[Dict]):
        """生成所有输出（可视化视频+CSV+结果窗口）"""
        try:
            self.generate_visualized_video(frame_data_list, joint_data_list)
            self.generate_joint_csv(joint_data_list, selected_joints)  # 传递选中关节
            statistics = self.calculate_motion_statistics(frame_data_list, joint_data_list, selected_joints)  # 传递选中关节
            if statistics:
                self.show_results_window(statistics, selected_joints)
            print("所有输出生成完成！")
        except Exception as e:
            print(f"输出生成失败：{str(e)}")
            raise