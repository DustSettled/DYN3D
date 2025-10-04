import torch
import torch.nn as nn
from torchvision import models
from torch import Tensor

class VGG16XYTZPredictor(nn.Module):
    def __init__(self, input_time_dim=1, output_dim=4):
        """
        基于 VGG16 的单关键帧时空预测器（输出下一时刻 xyzt）
        参数：
            input_time_dim: 时间戳输入维度（默认1，单值时间戳）
            output_dim: 输出维度（默认4，x,y,z,t）
        """
        super(VGG16XYTZPredictor, self).__init__()
        
        # --------------------------
        # 1. 加载官方预训练 VGG16（特征提取核心）
        # --------------------------
        vgg16_pretrained = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg16_pretrained.features  # 卷积+池化特征提取层（输入需为 [B, 3, H, W]）
        
        # --------------------------
        # 2. 单关键帧时空特征编码模块（适配 [B,3] 空间坐标 + [B,1] 时间戳）
        # --------------------------
        # 时间戳编码器（单值时间戳 → 特征）
        self.time_encoder = nn.Sequential(
            nn.Linear(input_time_dim, 16),  # 时间戳映射到16维
            nn.ReLU(),
            nn.Unflatten(1, (16, 1, 1)),  # 转换为 [B, 16, 1, 1]（适配卷积输入）
            nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0)),  # 沿时间维度卷积
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化 → [B, 32, 1, 1]
            nn.Flatten(start_dim=1),  # 展平为 [B, 32]
            nn.Linear(32, 64)  # 映射到与空间特征同维度
        )
        
        # 空间坐标编码器（[B,3] → 特征）
        self.space_encoder = nn.Sequential(
            nn.Linear(3, 16),  # 3维坐标映射到16维
            nn.ReLU(),
            nn.Unflatten(1, (16, 1, 1)),  # 转换为 [B, 16, 1, 1]（适配卷积输入）
            nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0)),  # 沿空间维度卷积
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化 → [B, 32, 1, 1]
            nn.Flatten(start_dim=1),  # 展平为 [B, 32]
            nn.Linear(32, 64)  # 映射到与时间特征同维度
        )
        
        # --------------------------
        # 3. 时空特征融合与输出头
        # --------------------------
        # 融合时间与空间特征（64+64 → 128 → 256 → 4）
        self.fusion_classifier = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)  # 输出 [B, 4]（x,y,z,t）
        )
        
        # 冻结预训练权重（可选，微调时注释）
        # for param in self.features.parameters():
        #     param.requires_grad = False
            
    def forward(self, spatial_xyz: Tensor, base_times: Tensor) -> Tensor:
        """
        输入：
            spatial_xyz: 当前关键帧空间坐标 [B, 3]（B批量，3坐标）
            base_times: 当前关键帧时间戳 [B, 1]（B批量，单值时间）
        输出：
            pred_xyzt: 下一时刻预测坐标 [B, 4]（x,y,z,t）
        """
        # --------------------------
        # 步骤1：编码时间戳特征
        # --------------------------
        time_feat = self.time_encoder(base_times)  # [B, 64]
        
        # --------------------------
        # 步骤2：编码空间坐标特征
        # --------------------------
        space_feat = self.space_encoder(spatial_xyz)  # [B, 64]
        
        # --------------------------
        # 步骤3：融合时空特征并预测
        # --------------------------
        fused_feat = torch.cat([time_feat, space_feat], dim=-1)  # [B, 128]
        pred_xyzt = self.fusion_classifier(fused_feat)  # [B, 4]
        
        return pred_xyzt