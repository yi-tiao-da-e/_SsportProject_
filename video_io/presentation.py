import cv2
import pandas as pd
import os
from typing import List
from dataclasses import asdict
from config.settings import app_config
from video_io.data_classes import FrameData  # 从独立文件导入FrameData
from recog.data_classes import JointData  # 从独立文件导入JointData

class Presentation:
    """负责生成可视化视频与数据表格的模块"""
    def __init__(self):
        # 从配置中获取输出路径
        self.output_video_path = app_config.video.output_video_path
        self.output_csv_path = app_config.video.output_csv_path
        # 确保输出目录存在（递归创建）
        os.makedirs(os.path.dirname(self.output_video_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
        # 可视化配置
        self.joint_visual_config = {
            "elbow": {
                "color": (0, 0, 255),  # BGR：红色（肘关节）
                "thickness": 2,
                "points": ("shoulder", "elbow", "wrist")  # 肩→肘→腕
            },
            "knee": {
                "color": (255, 0, 0),  # BGR：蓝色（膝关节）
                "thickness": 2,
                "points": ("hip", "knee", "ankle")  # 髋→膝→踝
            }
        }
        self.text_config = {
            "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
            "fontScale": 0.5,
            "color": (255, 255, 255),  # 白色文本
            "thickness": 1,
            "line_type": cv2.LINE_AA
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
            # 绘制连线（修复：正确连接点）
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

    def _draw_joint_angles(self, frame: cv2.Mat, joint_data: JointData):
        """在帧左上角绘制关节角度文本"""
        text_pos = (10, 30)
        line_spacing = 20

        # 绘制左肘角度
        if joint_data.left_elbow_angle is not None:
            text = f"Left Elbow: {joint_data.left_elbow_angle:.1f}°"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)  # 修复：正确更新y坐标

        # 绘制右肘角度
        if joint_data.right_elbow_angle is not None:
            text = f"Right Elbow: {joint_data.right_elbow_angle:.1f}°"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)

        # 绘制左膝角度
        if joint_data.left_knee_angle is not None:
            text = f"Left Knee: {joint_data.left_knee_angle:.1f}°"
            cv2.putText(frame, text, text_pos, **self.text_config)
            text_pos = (text_pos[0], text_pos[1] + line_spacing)

        # 绘制右膝角度
        if joint_data.right_knee_angle is not None:
            text = f"Right Knee: {joint_data.right_knee_angle:.1f}°"
            cv2.putText(frame, text, text_pos, **self.text_config)

    def generate_output(self, frame_data_list: List[FrameData], joint_data_list: List[JointData]):
        """生成所有输出(可视化视频+CSV表格)"""
        print("开始生成输出...")
        try:
            self.generate_visualized_video(frame_data_list, joint_data_list)
            self.generate_joint_csv(joint_data_list)
            print("所有输出生成完成！")
        except Exception as e:
            print(f"输出生成失败：{str(e)}")
            raise  # 重新抛出异常，让上层处理