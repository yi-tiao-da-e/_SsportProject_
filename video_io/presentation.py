import cv2
import pandas as pd
import os
from typing import List ,Set
from dataclasses import asdict
from config.settings import app_config
from video_io.data_classes import FrameData  # 从独立文件导入FrameData
from recog.data_classes import JointData  # 从独立文件导入JointData
import tkinter as tk
from tkinter import messagebox

class Presentation:
    """负责生成可视化视频与数据表格的模块"""
    def __init__(self, selected_joints: Set[str] = None):
        # 从配置中获取输出路径
        self.output_video_path = app_config.video.output_video_path
        self.output_csv_path = app_config.video.output_csv_path
        # 确保输出目录存在（递归创建）
        os.makedirs(os.path.dirname(self.output_video_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
        
        # 选择的关节部位
        self.selected_joints = selected_joints
        # 可视化配置
        self.joint_visual_config = {
        "elbow": {
            "color": (0, 0, 255),  # 红色
            "thickness": 2,
            "points": ("shoulder", "elbow", "wrist")
        },
        "knee": {
            "color": (255, 0, 0),  # 蓝色
            "thickness": 2,
            "points": ("hip", "knee", "ankle")
        },
        # 肩颈连线（左/右肩到颈椎）
        "shoulder_neck": {
            "color": (255, 255, 0),  # 青色
            "thickness": 2,
            "points": ("shoulder", "neck")  # 肩→颈椎
        },
        # 髋部连线（左/右髋到髋部中心）
        "hip_center": {
            "color": (255, 0, 255),  # 紫色
            "thickness": 2,
            "points": ("hip", "hip_center")  # 髋→髋部中心
        },
        # 脊柱连线（颈椎到髋部中心）
        "spine": {
            "color": (0, 255, 0),  # 绿色
            "thickness": 3,  # 加粗脊柱线
            "points": ("neck", "hip_center")  # 颈椎→髋部中心
        },
        # 手腕连线（从肘到腕）
        "wrist": {
            "color": (0, 255, 255),  # 黄色
            "thickness": 2,
            "points": ("elbow", "wrist")
        },
        # 脚踝连线（从膝到踝）
        "ankle": {
            "color": (255, 255, 0),  # 青色
            "thickness": 2,
            "points": ("knee", "ankle")
        }
    }
        # 文本配置 - 确保这个属性正确初始化
        self.text_config = {
            "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
            "fontScale": 0.5,
            "color": (255, 255, 255),  # 白色文本
            "thickness": 1,
            "lineType": cv2.LINE_AA
        }

    def generate_visualized_video(self, frame_data_list: List[FrameData], joint_data_list: List[JointData]):
        """生成叠加关节连线与角度的视频"""
        if not frame_data_list or not joint_data_list:
            raise ValueError("原始帧或关节数据为空")
        if len(frame_data_list) != len(joint_data_list):
            raise ValueError("原始帧与关节数据数量不匹配")

        # 从第一帧获取视频参数（分辨率、帧率）
        first_frame = frame_data_list[0]  # 修复：取列表第一个元素
        frame_height, frame_width = first_frame.frame.shape[:2]  # 修复：获取宽高的正确方式
        fps = first_frame.fps

        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*app_config.video.output_video_codec)
        video_writer = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )

        # 遍历每帧，叠加可视化元素
        for frame_data, joint_data in zip(frame_data_list, joint_data_list):
            frame = frame_data.frame.copy()
            self._draw_joint_lines(frame, joint_data, frame_width, frame_height)
            self._draw_joint_angles(frame, joint_data)
            video_writer.write(frame)

        video_writer.release()
        print(f"可视化视频已保存至：{self.output_video_path}")

    def generate_joint_csv(self, joint_data_list: List[JointData]):
        """将关节数据列表转换为CSV文件"""
        if not joint_data_list:
            raise ValueError("关节数据为空")
        df = pd.DataFrame([asdict(jd) for jd in joint_data_list])
        df.to_csv(self.output_csv_path, index=False)
        print(f"关节数据CSV已保存至：{self.output_csv_path}")

    def _draw_joint_lines(self, frame: cv2.Mat, joint_data: JointData, frame_width: int, frame_height: int):
        """在帧上绘制关节连线"""

        # 检查是否应该绘制某个连线
        def should_draw(joint_keys):
            if self.selected_joints is None:
                return True
            return any(key in self.selected_joints for key in joint_keys)
        
        # 处理肘关节（左/右）
        for side in ["left", "right"]:
            config = self.joint_visual_config["elbow"]
            points = []
            for point in config["points"]:
                x_field = f"{side}_{point}_x"
                y_field = f"{side}_{point}_y"
                x = getattr(joint_data, x_field, None)
                y = getattr(joint_data, y_field, None)
                if x is None or y is None:
                    break
                pixel_x = int(x * frame_width)
                pixel_y = int(y * frame_height)
                points.append((pixel_x, pixel_y))
            # 绘制连线
            if len(points) == 3:
                cv2.line(frame, points[0], points[1], config["color"], config["thickness"])  # 肩→肘
                cv2.line(frame, points[1], points[2], config["color"], config["thickness"])  # 肘→腕

        # 处理膝关节（左/右）
        for side in ["left", "right"]:
            config = self.joint_visual_config["knee"]
            points = []
            for point in config["points"]:
                x_field = f"{side}_{point}_x"
                y_field = f"{side}_{point}_y"
                x = getattr(joint_data, x_field, None)
                y = getattr(joint_data, y_field, None)
                if x is None or y is None:
                    break
                pixel_x = int(x * frame_width)
                pixel_y = int(y * frame_height)
                points.append((pixel_x, pixel_y))
            if len(points) == 3:
                cv2.line(frame, points[0], points[1], config["color"], config["thickness"])  # 髋→膝
                cv2.line(frame, points[1], points[2], config["color"], config["thickness"])  # 膝→踝

        # 新增：绘制肩颈连线（左/右肩到颈椎）
        for side in ["left", "right"]:
            config = self.joint_visual_config["shoulder_neck"]
            points = []
            
            # 肩点
            shoulder_x = getattr(joint_data, f"{side}_shoulder_x", None)
            shoulder_y = getattr(joint_data, f"{side}_shoulder_y", None)
            if shoulder_x is not None and shoulder_y is not None:
                points.append((int(shoulder_x * frame_width), int(shoulder_y * frame_height)))
            
            # 颈椎点
            neck_x = getattr(joint_data, "neck_x", None)
            neck_y = getattr(joint_data, "neck_y", None)
            if neck_x is not None and neck_y is not None:
                points.append((int(neck_x * frame_width), int(neck_y * frame_height)))
            
            # 绘制肩颈连线
            if len(points) == 2:
                cv2.line(frame, points[0], points[1], config["color"], config["thickness"])

        # 新增：绘制髋部中心连线（左/右髋到髋部中心）
        for side in ["left", "right"]:
            config = self.joint_visual_config["hip_center"]
            points = []
            
            # 髋点
            hip_x = getattr(joint_data, f"{side}_hip_x", None)
            hip_y = getattr(joint_data, f"{side}_hip_y", None)
            if hip_x is not None and hip_y is not None:
                points.append((int(hip_x * frame_width), int(hip_y * frame_height)))
            
            # 髋部中心点
            hip_center_x = getattr(joint_data, "hip_center_x", None)
            hip_center_y = getattr(joint_data, "hip_center_y", None)
            if hip_center_x is not None and hip_center_y is not None:
                points.append((int(hip_center_x * frame_width), int(hip_center_y * frame_height)))
            
            # 绘制髋部中心连线
            if len(points) == 2:
                cv2.line(frame, points[0], points[1], config["color"], config["thickness"])

        # 新增：绘制脊柱连线（颈椎到髋部中心）
        config = self.joint_visual_config["spine"]
        points = []
        
        # 颈椎点
        neck_x = getattr(joint_data, "neck_x", None)
        neck_y = getattr(joint_data, "neck_y", None)
        if neck_x is not None and neck_y is not None:
            points.append((int(neck_x * frame_width), int(neck_y * frame_height)))
        
        # 髋部中心点
        hip_center_x = getattr(joint_data, "hip_center_x", None)
        hip_center_y = getattr(joint_data, "hip_center_y", None)
        if hip_center_x is not None and hip_center_y is not None:
            points.append((int(hip_center_x * frame_width), int(hip_center_y * frame_height)))
        
        # 绘制脊柱连线
        if len(points) == 2:
            cv2.line(frame, points[0], points[1], config["color"], config["thickness"])

        # 绘制手腕连线
        for side in ["left", "right"]:
            config = self.joint_visual_config["wrist"]
            points = []
            for point in config["points"]:
                x_field = f"{side}_{point}_x"
                y_field = f"{side}_{point}_y"
                x = getattr(joint_data, x_field, None)
                y = getattr(joint_data, y_field, None)
                if x is None or y is None:
                    break
                pixel_x = int(x * frame_width)
                pixel_y = int(y * frame_height)
                points.append((pixel_x, pixel_y))
            if len(points) == 2:
                cv2.line(frame, points[0], points[1], config["color"], config["thickness"])

        # 绘制脚踝连线
        for side in ["left", "right"]:
            config = self.joint_visual_config["ankle"]
            points = []
            for point in config["points"]:
                x_field = f"{side}_{point}_x"
                y_field = f"{side}_{point}_y"
                x = getattr(joint_data, x_field, None)
                y = getattr(joint_data, y_field, None)
                if x is None or y is None:
                    break
                pixel_x = int(x * frame_width)
                pixel_y = int(y * frame_height)
                points.append((pixel_x, pixel_y))
            if len(points) == 2:
                cv2.line(frame, points[0], points[1], config["color"], config["thickness"])

        # 绘制关节点（在所有关节位置绘制小圆点）
        self._draw_joint_points(frame, joint_data, frame_width, frame_height)

    def _draw_joint_points(self, frame: cv2.Mat, joint_data: JointData, frame_width: int, frame_height: int):
        """在所有关节位置绘制小圆点"""
        joint_points = [
            # 上半身
            ("left_shoulder", (0, 255, 0)),    # 绿色
            ("left_elbow", (0, 255, 0)),       # 绿色
            ("left_wrist", (0, 255, 0)),       # 绿色
            ("right_shoulder", (0, 255, 0)),   # 绿色
            ("right_elbow", (0, 255, 0)),      # 绿色
            ("right_wrist", (0, 255, 0)),      # 绿色
            # 躯干虚拟节点
            ("neck", (255, 255, 0)),           # 青色（颈椎虚拟节点）
            ("hip_center", (255, 0, 255)),     # 紫色（髋部中心虚拟节点）
            # 下半身
            ("left_hip", (255, 0, 0)),         # 蓝色
            ("left_knee", (255, 0, 0)),        # 蓝色
            ("left_ankle", (255, 0, 0)),       # 蓝色
            ("right_hip", (255, 0, 0)),        # 蓝色
            ("right_knee", (255, 0, 0)),       # 蓝色
            ("right_ankle", (255, 0, 0))       # 蓝色
        ]
        
        for joint_name, color in joint_points:
            x_field = f"{joint_name}_x"
            y_field = f"{joint_name}_y"
            x = getattr(joint_data, x_field, None)
            y = getattr(joint_data, y_field, None)
            
            if x is not None and y is not None:
                pixel_x = int(x * frame_width)
                pixel_y = int(y * frame_height)
                # 绘制关节点圆点
                cv2.circle(frame, (pixel_x, pixel_y), 4, color, -1)  # 实心圆点，半径4像素

    def _draw_joint_angles(self, frame: cv2.Mat, joint_data: JointData):
        """在帧上绘制关节角度文本（扩展新增部位）"""
        text_pos = (10, 30)
        line_spacing = 20

        # 绘制左肘角度
        if joint_data.left_elbow_angle is not None:
            text = f"Left Elbow: {joint_data.left_elbow_angle:.1f}ang"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing) 

        # 绘制右肘角度
        if joint_data.right_elbow_angle is not None:
            text = f"Right Elbow: {joint_data.right_elbow_angle:.1f}ang"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)

        # 绘制左膝角度
        if joint_data.left_knee_angle is not None:
            text = f"Left Knee: {joint_data.left_knee_angle:.1f}ang"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)

        # 绘制右膝角度
        if joint_data.right_knee_angle is not None:
            text = f"Right Knee: {joint_data.right_knee_angle:.1f}ang"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)

        # 新增：腰椎角度
        if joint_data.spine_angle is not None:
            text = f"Spine: {joint_data.spine_angle:.1f}ang"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)

        # 新增：髋部角度
        if joint_data.left_hip_angle is not None:
            text = f"Left Hip: {joint_data.left_hip_angle:.1f}ang"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)
            
        if joint_data.right_hip_angle is not None:
            text = f"Right Hip: {joint_data.right_hip_angle:.1f}ang"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)

        # 新增：脚踝角度
        if joint_data.left_ankle_angle is not None:
            text = f"Left Ankle: {joint_data.left_ankle_angle:.1f}ang"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)
            
        if joint_data.right_ankle_angle is not None:
            text = f"Right Ankle: {joint_data.right_ankle_angle:.1f}ang"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)
            
            # 绘制左肘角速度
        if joint_data.left_elbow_velocity is not None:
            # 格式化为一位小数，添加单位°/s
            text = f"L Elbow Vel: {joint_data.left_elbow_velocity:.1f}ang/s"  
            cv2.putText(frame, text, text_pos, **self.text_config)
            # 更新到下一行位置
            text_pos = (text_pos[0], text_pos[1] + line_spacing)  

        # 绘制右肘角速度
        if joint_data.right_elbow_velocity is not None:
            text = f"R Elbow Vel: {joint_data.right_elbow_velocity:.1f}ang/s"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)

        # 绘制左膝角速度
        if joint_data.left_knee_velocity is not None:
            text = f"L Knee Vel: {joint_data.left_knee_velocity:.1f}ang/s"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)

        # 绘制右膝角速度
        if joint_data.right_knee_velocity is not None:
            text = f"R Knee Vel: {joint_data.right_knee_velocity:.1f}ang/s"
            cv2.putText(frame, text, text_pos, **self.text_config)
        
        # 新增：腰椎角速度
        if joint_data.spine_velocity is not None:
            text = f"Spine Vel: {joint_data.spine_velocity:.1f}ang/s"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)

        # 新增：髋部角速度
        if joint_data.left_hip_velocity is not None:
            text = f"L Hip Vel: {joint_data.left_hip_velocity:.1f}ang/s"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)
            
        if joint_data.right_hip_velocity is not None:
            text = f"R Hip Vel: {joint_data.right_hip_velocity:.1f}ang/s"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)

        # 新增：脚踝角速度
        if joint_data.left_ankle_velocity is not None:
            text = f"L Ankle Vel: {joint_data.left_ankle_velocity:.1f}ang/s"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)
            
        if joint_data.right_ankle_velocity is not None:
            text = f"R Ankle Vel: {joint_data.right_ankle_velocity:.1f}ang/s"
            cv2.putText(frame, text, text_pos, **self.text_config)
            
    def calculate_motion_statistics(self, frame_data_list: List[FrameData], joint_data_list: List[JointData]):
        """计算包括新增部位在内的运动统计信息"""
        if not frame_data_list or not joint_data_list:
            return None
        
        # 计算运动时长
        start_time = frame_data_list[0].timestamp
        end_time = frame_data_list[-1].timestamp
        duration = end_time - start_time

        # 扩展最大关节角速度统计 - 包含所有可能的关节
        max_velocities = {
            # 上半身
            "left_elbow": float('-inf'),
            "right_elbow": float('-inf'),
            "left_shoulder": float('-inf'),
            "right_shoulder": float('-inf'),
            "left_wrist": float('-inf'),
            "right_wrist": float('-inf'),
            # 下半身
            "left_knee": float('-inf'),
            "right_knee": float('-inf'),
            "left_hip": float('-inf'),
            "right_hip": float('-inf'),
            "left_ankle": float('-inf'),
            "right_ankle": float('-inf'),
            # 躯干
            "spine": float('-inf'),
            "neck": float('-inf'),
            "hip_center": float('-inf')
        }

        # 遍历所有关节数据
        for joint_data in joint_data_list:
            # 原有统计代码...
            
            # 肘部角速度
            if joint_data.left_elbow_velocity is not None:
                if joint_data.left_elbow_velocity > max_velocities["left_elbow"]:
                    max_velocities["left_elbow"] = joint_data.left_elbow_velocity
            if joint_data.right_elbow_velocity is not None:
                if joint_data.right_elbow_velocity > max_velocities["right_elbow"]:
                    max_velocities["right_elbow"] = joint_data.right_elbow_velocity
            
            # 膝部角速度
            if joint_data.left_knee_velocity is not None:
                if joint_data.left_knee_velocity > max_velocities["left_knee"]:
                    max_velocities["left_knee"] = joint_data.left_knee_velocity
            if joint_data.right_knee_velocity is not None:
                if joint_data.right_knee_velocity > max_velocities["right_knee"]:
                    max_velocities["right_knee"] = joint_data.right_knee_velocity
            
            # 新增：脊柱角速度
            if joint_data.spine_velocity is not None:
                if joint_data.spine_velocity > max_velocities["spine"]:
                    max_velocities["spine"] = joint_data.spine_velocity
                    
            # 新增：髋部角速度
            if joint_data.left_hip_velocity is not None:
                if joint_data.left_hip_velocity > max_velocities["left_hip"]:
                    max_velocities["left_hip"] = joint_data.left_hip_velocity
            if joint_data.right_hip_velocity is not None:
                if joint_data.right_hip_velocity > max_velocities["right_hip"]:
                    max_velocities["right_hip"] = joint_data.right_hip_velocity
                    
            # 新增：脚踝角速度
            if joint_data.left_ankle_velocity is not None:
                if joint_data.left_ankle_velocity > max_velocities["left_ankle"]:
                    max_velocities["left_ankle"] = joint_data.left_ankle_velocity
            if joint_data.right_ankle_velocity is not None:
                if joint_data.right_ankle_velocity > max_velocities["right_ankle"]:
                    max_velocities["right_ankle"] = joint_data.right_ankle_velocity

        # 处理无有效值的情况
        for joint in max_velocities:
            if max_velocities[joint] == float('-inf'):
                max_velocities[joint] = 0.0

        return {
            "duration": duration,
            "max_velocities": max_velocities
        }
        
    def show_results_window(self, statistics):
        """显示结果窗口（显示所有关节节点的最大角速度）"""
        print("正在显示结果窗口...")
        if not statistics:
            print("无统计信息可显示")
            return
        try:    
            # 创建结果窗口
            root = tk.Tk()
            root.title("运动分析结果")
            root.geometry("500x600")  # 增加窗口高度以容纳更多内容
            root.resizable(False, False)
        
            # 设置窗口居中
            root.update_idletasks()
            x = (root.winfo_screenwidth() // 2) - (500 // 2)
            y = (root.winfo_screenheight() // 2) - (600 // 2)
            root.geometry(f"500x600+{x}+{y}")
        
            # 添加标题
            title_label = tk.Label(root, text="运动分析结果", font=("Arial", 16, "bold"))
            title_label.pack(pady=20)
        
            # 添加运动时长
            duration_label = tk.Label(
                root, 
                text=f"运动时长: {statistics['duration']:.2f} 秒", 
                font=("Arial", 14)
            )
            duration_label.pack(pady=10)
        
            # ==================== 所有关节最大角速度显示 ====================
            # 添加角速度标题
            velocity_title = tk.Label(
                root, 
                text="最大关节角速度", 
                font=("Arial", 12, "bold")
            )
            velocity_title.pack(pady=(20, 10))
        
            # 创建主框架用于滚动
            main_frame = tk.Frame(root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
            # 创建画布和滚动条
            canvas = tk.Canvas(main_frame, height=400)
            scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)
        
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
        
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
        
            # 获取最大角速度数据
            max_velocities = statistics.get("max_velocities", {})
            
            # 定义所有关节的中文名称映射
            joint_names = {
                "left_elbow": "左肘",
                "right_elbow": "右肘",
                "left_shoulder": "左肩",
                "right_shoulder": "右肩",
                "left_wrist": "左腕",
                "right_wrist": "右腕",
                "left_knee": "左膝",
                "right_knee": "右膝",
                "left_hip": "左髋",
                "right_hip": "右髋",
                "left_ankle": "左踝",
                "right_ankle": "右踝",
                "spine": "脊柱",
                "neck": "颈椎",
                "hip_center": "髋部中心"
            }
        
            # 按类别分组显示
            categories = {
                "上半身": ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist"],
                "下半身": ["left_knee", "right_knee", "left_hip", "right_hip", "left_ankle", "right_ankle"],
                "躯干": ["spine", "neck", "hip_center"]
            }
        
            row = 0
            for category, joints in categories.items():
                # 类别标签
                category_label = tk.Label(
                    scrollable_frame, 
                    text=category, 
                    font=("Arial", 11, "bold"),
                    bg='lightgray',
                    relief=tk.RAISED,
                    bd=1
                )
                category_label.grid(row=row, column=0, columnspan=2, sticky="we", pady=(10, 5), padx=5)
                row += 1
                
                # 该类别下的关节
                for joint_key in joints:
                    chinese_name = joint_names.get(joint_key, joint_key)
                    
                    # 检查该关节是否被选中
                    is_selected = self.selected_joints is None or joint_key in self.selected_joints
                    
                    if is_selected:
                        # 选中的关节：显示角速度值
                        velocity_value = max_velocities.get(joint_key, 0.0)
                        text = f"{chinese_name}: {velocity_value:.1f} °/s"
                        color = "black"  # 正常颜色
                    else:
                        # 未选中的关节：显示"未选中"
                        text = f"{chinese_name}: 未选中"
                        color = "gray"  # 灰色表示未选中
                    
                    joint_label = tk.Label(
                        scrollable_frame,
                        text=text,
                        font=("Arial", 10),
                        fg=color
                    )
                    joint_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=20, pady=2)
                    row += 1
        
            # 打包画布和滚动条
            canvas.grid(row=0, column=0, sticky="nsew")
            scrollbar.grid(row=0, column=1, sticky="ns")
            main_frame.grid_rowconfigure(0, weight=1)
            main_frame.grid_columnconfigure(0, weight=1)
            # =============================================================
        
            # 添加确定按钮
            ok_button = tk.Button(root, text="确定", command=root.destroy, width=10, height=2)
            ok_button.pack(pady=20)
        
            print("结果窗口已创建，开始主循环")
            root.mainloop()
            print("结果窗口已关闭")
        
        except Exception as e:
            print(f"显示结果窗口时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def generate_output(self, frame_data_list: List[FrameData], joint_data_list: List[JointData]):
        """生成所有输出(可视化视频+CSV表格)"""
        print("开始生成输出...")
        try:
            self.generate_visualized_video(frame_data_list, joint_data_list)
            self.generate_joint_csv(joint_data_list)
            print("所有输出生成完成！")
            
            # 计算统计信息并显示结果窗口
            statistics = self.calculate_motion_statistics(frame_data_list, joint_data_list)
            print(f"统计信息: {statistics}")  # 调试信息
        
            if statistics:
                print("准备显示结果窗口...")
                # 直接显示结果窗口（阻塞主线程，直到窗口关闭）
                self.show_results_window(statistics)
            else:
                print("无统计信息可显示")
            print("所有输出生成完成！")
        except Exception as e:
            print(f"输出生成失败：{str(e)}")
            import traceback
            traceback.print_exc()
            
            raise  # 重新抛出异常，让上层处理