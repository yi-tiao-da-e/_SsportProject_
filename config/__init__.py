class MediapipeConfig:
    def __init__(self, model_complexity=2, min_detection_confidence=0.7, 
                 min_tracking_confidence=0.5, enable_segmentation=False, 
                 smooth_landmarks=True):
        # 使用传入的参数值，如果没有提供则使用默认值
        self.model_complexity = model_complexity #模型精度0、1、2
        self.min_detection_confidence = min_detection_confidence #检测置信度
        self.min_tracking_confidence = min_tracking_confidence   #跟踪置信度
        self.enable_segmentation = enable_segmentation  #姿态分割
        self.smooth_landmarks = smooth_landmarks  #关节平滑

class IOConfig:
    def __init__(self,InSourceAdr,OutSourceAdr):
        self.InSourceAdr = InSourceAdr
        self.OutSourceAdr = OutSourceAdr