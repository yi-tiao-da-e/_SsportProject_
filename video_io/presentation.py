import cv2
import pandas as pd
import os
from typing import List, Optional
from dataclasses import asdict
from config.settings import app_config
from video_io.input_handler import FrameData
from recog.med_processor import JointData  # 需包含关节点坐标

class Presentation:
    """负责生成可视化视频与数据表格的模块"""
    def __init__(self):
        # 从配置中获取输出路径
        self.output_video_path = app_config.video.output_video_path
        self.output_csv_path = app_config.video.output_csv_path
        # 确保输出目录存在（递归创建）
        os.makedirs(os.path.dirname(self.output_video_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
        # 可视化配置（可从settings中提取，此处简化）
        self.joint_visual_config = {
            "elbow": {
                "color": (0, 0, 255),  # BGR：红色（肘关节）
                "thickness": 2,
                "points": ("shoulder", "elbow", "wrist")  # 关节点顺序（肩→肘→腕）
            },
            "knee": {
                "color": (255, 0, 0),  # BGR：蓝色（膝关节）
                "thickness": 2,
                "points": ("hip", "knee", "ankle")  # 关节点顺序（髋→膝→踝）
            }
        }
        self.text_config = {
            "font": cv2.FONT_HERSHEY_SIMPLEX,
            "font_scale": 0.5,
            "color": (255, 255, 255),  # 白色文本
            "thickness": 1,
            "line_type": cv2.LINE_AA
        }



def generate_visualized_video(self, frame_data_list: List[FrameData], joint_data_list: List[JointData]):
    """
    生成叠加关节连线与角度的视频
    :param frame_data_list: 原始帧数据列表（FrameData）
    :param joint_data_list: 关节数据列表（JointData，需包含关节点坐标）
    """
    if not frame_data_list or not joint_data_list:
        raise ValueError("原始帧或关节数据为空")
    if len(frame_data_list) != len(joint_data_list):
        raise ValueError("原始帧与关节数据数量不匹配")

    # 从第一帧获取视频参数（分辨率、帧率）
    first_frame = frame_data_list
    frame_width = first_frame.frame.shape
    frame_height = first_frame.frame.shape
    fps = first_frame.fps

    # 初始化视频写入器（MP4格式）
    fourcc = cv2.VideoWriter_fourcc(*app_config.video.output_video_codec)  # 从settings取编码（如'mp4v'）
    video_writer = cv2.VideoWriter(
        self.output_video_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    # 遍历每帧，叠加可视化元素
    for frame_data, joint_data in zip(frame_data_list, joint_data_list):
        frame = frame_data.frame.copy()  # 复制原始帧，避免修改源数据

        # 1. 叠加关节连线（肘关节、膝关节）
        self._draw_joint_lines(frame, joint_data, frame_width, frame_height)

        # 2. 叠加关节角度文本（左上角）
        self._draw_joint_angles(frame, joint_data)

        # 3. 写入输出视频
        video_writer.write(frame)

    # 释放资源
    video_writer.release()
    print(f"可视化视频已保存至：{self.output_video_path}")

def generate_joint_csv(self, joint_data_list: List[JointData]):
    """
    将关节数据列表转换为CSV文件
    :param joint_data_list: 关节数据列表（JointData）
    """
    if not joint_data_list:
        raise ValueError("关节数据为空")

    # 将JointData列表转换为DataFrame（使用dataclass的asdict方法）
    df = pd.DataFrame([asdict(jd) for jd in joint_data_list])

    # 保存为CSV（不含索引）
    df.to_csv(self.output_csv_path, index=False)
    print(f"关节数据CSV已保存至：{self.output_csv_path}")

def _draw_joint_lines(self, frame: cv2.Mat, joint_data: JointData, frame_width: int, frame_height: int):
    """
    在帧上绘制关节连线（肘关节、膝关节）
    :param frame: 当前帧（BGR格式）
    :param joint_data: 当前帧的关节数据（需包含关节点坐标）
    :param frame_width: 帧宽度（像素）
    :param frame_height: 帧高度（像素）
    """
    # 处理肘关节（左/右）
    for side in ["left", "right"]:
        # 获取肘关节配置（颜色、厚度、点顺序）
        config = self.joint_visual_config["elbow"]
        # 提取关节点坐标（归一化→像素）
        points = []
        for point in config["points"]:
            # 构造JointData中的字段名（如left_shoulder_x）
            x_field = f"{side}_{point}_x"
            y_field = f"{side}_{point}_y"
            # 获取坐标（若可见性低，跳过）
            x = getattr(joint_data, x_field, None)
            y = getattr(joint_data, y_field, None)
            if x is None or y is None:
                break
            # 转换为像素坐标（归一化值×帧尺寸）
            pixel_x = int(x * frame_width)
            pixel_y = int(y * frame_height)
            points.append((pixel_x, pixel_y))
        # 绘制连线（至少3个点才画）
        if len(points) == 3:
            cv2.line(frame, points, points, config["color"], config["thickness"])
            cv2.line(frame, points, points, config["color"], config["thickness"])

    # 处理膝关节（左/右），逻辑与肘关节一致
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
            cv2.line(frame, points, points, config["color"], config["thickness"])
            cv2.line(frame, points, points, config["color"], config["thickness"])

def _draw_joint_angles(self, frame: cv2.Mat, joint_data: JointData):
    """
    在帧左上角绘制关节角度文本（左/右肘、左/右膝）
    :param frame: 当前帧（BGR格式）
    :param joint_data: 当前帧的关节数据
    """
    text_pos = (10, 30)  # 初始位置（左上角）
    line_spacing = 20     # 行间距（像素）

    # 绘制左肘角度
    if joint_data.left_elbow_angle is not None:
        text = f"Left Elbow: {joint_data.left_elbow_angle:.1f}°"
        cv2.putText(
            frame,
            text,
            text_pos,
            self.text_config["font"],
            self.text_config["font_scale"],
            self.text_config["color"],
            self.text_config["thickness"],
            self.text_config["line_type"]
        )
        text_pos = (text_pos, text_pos + line_spacing)

    # 绘制右肘角度
    if joint_data.right_elbow_angle is not None:
        text = f"Right Elbow: {joint_data.right_elbow_angle:.1f}°"
        cv2.putText(frame, text, text_pos, **self.text_config)
        text_pos = (text_pos, text_pos + line_spacing)

    # 绘制左膝角度
    if joint_data.left_knee_angle is not None:
        text = f"Left Knee: {joint_data.left_knee_angle:.1f}°"
        cv2.putText(frame, text, text_pos, **self.text_config)
        text_pos = (text_pos, text_pos + line_spacing)

    # 绘制右膝角度
    if joint_data.right_knee_angle is not None:
        text = f"Right Knee: {joint_data.right_knee_angle:.1f}°"
        cv2.putText(frame, text, text_pos, **self.text_config)


def generate_output(self, frame_data_list: List[FrameData], joint_data_list: List[JointData]):
    """
    生成所有输出(可视化视频+CSV表格)
    :param frame_data_list: 原始帧数据列表
    :param joint_data_list: 关节数据列表
    """
    try:
        self.generate_visualized_video(frame_data_list, joint_data_list)
        self.generate_joint_csv(joint_data_list)
        print("所有输出生成完成！")
    except Exception as e:
        raise RuntimeError(f"输出生成失败：{str(e)}") from e

def generate_output(self, frame_data_list: List[FrameData], joint_data_list: List[JointData]):
    print("开始生成输出...")  # 新增
    try:
        self.generate_visualized_video(frame_data_list, joint_data_list)
        self.generate_joint_csv(joint_data_list)
        print("输出完成！")  # 新增
    except Exception as e:
        print(f"输出生成失败：{str(e)}")  # 新增：打印异常
        raise