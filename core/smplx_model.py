# core/smplx_model.py
import numpy as np
import torch
import smplx
import logging
import time
from typing import Optional, List, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SMPLXModelHandler:
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        优化后的SMPLX模型处理器
        
        参数:
            model_path: SMPLX模型文件路径
            device: 指定设备，如'cuda'或'cpu'，默认自动选择
        """
        self.model_path = r'F:\_SsportProject_\models\smplx\SMPLX_NEUTRAL.npz'
        self.device = self._get_device(device)
        
        # 初始化模型
        self.smplx_model = None
        self.betas = None
        self.expression = None
        self.global_orient = None
        self.body_pose = None
        self.initialized = False
        
        # 初始化连接关系（固定）
        self.connections = self._get_default_connections()
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """获取最佳计算设备"""
        if device:
            return torch.device(device)
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def initialize(self):
        """延迟初始化模型，避免在__init__中阻塞主线程"""
        if self.initialized:
            return
        
        try:
            logger.info(f"正在加载SMPLX模型，路径: {self.model_path}")
            start_time = time.time()
            
            # 加载SMPLX模型
            self.smplx_model = smplx.create(
                model_path=self.model_path,
                model_type='smplx',
                gender='neutral',
                use_face_contour=False,
                num_betas=10,
                num_expression_coeffs=10,
                ext='npz',
                use_pca=False,
            ).to(self.device)
            
            # 初始化参数
            self.betas = torch.zeros([1, self.smplx_model.num_betas], 
                                    dtype=torch.float32, device=self.device)
            self.expression = torch.zeros([1, self.smplx_model.num_expression_coeffs], 
                                         dtype=torch.float32, device=self.device)
            self.global_orient = torch.zeros([1, 3], device=self.device)
            self.body_pose = torch.zeros([1, self.smplx_model.NUM_BODY_JOINTS * 3], 
                                        device=self.device)
            
            load_time = time.time() - start_time
            logger.info(f"SMPLX模型加载成功! 耗时: {load_time:.2f}秒")
            self.initialized = True
        except Exception as e:
            logger.error(f"无法加载SMPLX模型: {str(e)}")
            raise
    
    def _get_default_connections(self) -> List[Tuple[int, int]]:
        """获取默认的连接关系"""
        return [
            # 身体核心连接
            (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
            # 手臂连接
            (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),
            (13, 15), (15, 17), (17, 19), (19, 21), (21, 23),
            # 腿部连接
            (1, 7), (7, 9), (9, 11), (11, 13),
            (2, 8), (8, 10), (10, 12), (12, 14),
            # 面部连接 (简化)
            (24, 25), (25, 26), (26, 27), (27, 28),
            (24, 29), (29, 30), (30, 31), (31, 32)
        ]
    
    @staticmethod
    def map_mediapipe_to_smplx(mediapipe_points: np.ndarray) -> np.ndarray:
        """
        优化后的MediaPipe到SMPLX映射
        
        参数:
            mediapipe_points: (33, 3)或(33, 4)的numpy数组
        
        返回:
            smplx_points: (24, 3)的numpy数组
        """
        if mediapipe_points is None or mediapipe_points.size == 0:
            return np.zeros((24, 3))
        
        # 确保使用正确的维度
        mp = mediapipe_points[:, :3] if len(mediapipe_points.shape) > 1 else mediapipe_points.reshape(-1, 4)[:, :3]
        
        # 使用更稳定的计算方式
        smplx = np.zeros((24, 3))
        
        # 直接映射点
        direct_mapping = {
            23: 1, 24: 2, 25: 4, 26: 5, 
            27: 7, 28: 8, 31: 10, 32: 11,
            11: 16, 12: 17, 13: 18, 14: 19,
            15: 20, 16: 21
        }
        
        for mp_idx, smplx_idx in direct_mapping.items():
            if mp_idx < len(mp):
                smplx[smplx_idx] = mp[mp_idx]
        
        # 计算关键点
        if len(mp) > 24:
            pelvis = (mp[23] + mp[24]) / 2
            shoulder_center = (mp[11] + mp[12]) / 2
            head = mp[0]
            
            # 骨盆和脊柱
            smplx[0] = pelvis
            smplx[3] = pelvis * 0.67 + shoulder_center * 0.33  # 脊柱1
            smplx[6] = pelvis * 0.33 + shoulder_center * 0.67  # 脊柱2
            smplx[9] = shoulder_center  # 脊柱3
            
            # 头部和颈部
            neck = shoulder_center * 0.75 + head * 0.25
            smplx[12] = neck
            smplx[15] = head
            
            # 锁骨
            smplx[13] = (mp[11] + neck) / 2  # 左锁骨
            smplx[14] = (mp[12] + neck) / 2  # 右锁骨
        
        return smplx
    
    def fit_to_keypoints(self, keypoints_3d: np.ndarray) -> np.ndarray:
        """
        优化的关键点拟合方法，支持批量处理
        
        参数:
            keypoints_3d: MediaPipe检测到的3D关键点 (N, 33, 3)
            
        返回:
            拟合后的SMPLX关节位置 (N, 55, 3)
        """
        if not self.initialized:
            self.initialize()
        
        if keypoints_3d.size == 0:
            return np.array([])
        
        # 批量映射关键点
        start_time = time.time()
        smplx_points = np.array([self.map_mediapipe_to_smplx(frame) for frame in keypoints_3d])
        mapping_time = time.time() - start_time
        logger.debug(f"关键点映射完成，耗时: {mapping_time:.4f}秒")
        
        # 转换为Tensor
        keypoints_tensor = torch.tensor(smplx_points, dtype=torch.float32, device=self.device)
        num_frames = keypoints_tensor.shape[0]
        
        # 使用批处理
        batch_size = min(64, num_frames)  # 可调整的批处理大小
        all_joints = []
        
        for i in range(0, num_frames, batch_size):
            batch_end = min(i + batch_size, num_frames)
            batch = keypoints_tensor[i:batch_end]
            
            # 使用with torch.no_grad()减少内存使用
            with torch.no_grad():
                # 准备批量参数
                batch_size_current = batch.shape[0]
                global_orient = torch.zeros([batch_size_current, 3], device=self.device)
                body_pose = torch.zeros([batch_size_current, self.smplx_model.NUM_BODY_JOINTS * 3], 
                                       device=self.device)
                
                # 运行SMPLX前向传播
                output = self.smplx_model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=self.betas.expand(batch_size_current, -1),
                    expression=self.expression.expand(batch_size_current, -1),
                    return_verts=True,
                    return_full_pose=True
                )
                
                # 获取关节位置
                joints = output.joints.detach().cpu().numpy()
                
                # 替换身体关节
                joints[:, :24] = batch.cpu().numpy()
                all_joints.append(joints)
        
        # 合并结果
        full_joints = np.concatenate(all_joints, axis=0)
        total_time = time.time() - start_time
        logger.info(f"SMPLX拟合完成! 总帧数: {num_frames}, 耗时: {total_time:.2f}秒, 平均: {total_time/num_frames:.4f}秒/帧")
        
        return full_joints
    
    def get_connections(self):
        return self.connections
    
    def cleanup(self):
        """显式清理资源"""
        if self.initialized:
            del self.smplx_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            self.initialized = False
            logger.info("SMPLX模型资源已释放")
    
    def __del__(self):
        self.cleanup()