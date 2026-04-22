import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """
    GoogLeNet 核心 Inception 模块
    包含 4 个分支，最终在通道维度拼接
    """
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        # 分支 1: 1x1 卷积
        self.branch1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 分支 2: 1x1 降维 + 3x3 卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 分支 3: 1x1 降维 + 5x5 卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # 分支 4: 3x3 最大池化 + 1x1 卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 在通道维度（dim=1）拼接所有分支
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class AuxiliaryClassifier(nn.Module):
    """
    辅助分类器（训练时使用，测试时可丢弃）
    用于缓解梯度消失问题
    """
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    """
    完整 GoogLeNet（Inception V1）网络结构
    """
    def __init__(self, num_classes=1000, aux_logits=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits  # 是否使用辅助分类器

        # 前几层卷积与池化
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception 模块组
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # 辅助分类器（仅在训练时启用）
        if self.aux_logits:
            self.aux1 = AuxiliaryClassifier(512, num_classes)
            self.aux2 = AuxiliaryClassifier(528, num_classes)

        # 最终分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化到 1x1
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 前几层
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)

        # Inception 3a-3b
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Inception 4a-4e + 辅助分类器 1
        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1_out = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2_out = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # Inception 5a-5b
        x = self.inception5a(x)
        x = self.inception5b(x)

        # 最终分类
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        # 训练时返回主输出 + 两个辅助输出，测试时仅返回主输出
        if self.training and self.aux_logits:
            return x, aux1_out, aux2_out
        return x


# ------------------------------
# 测试代码：验证模型结构
# ------------------------------
if __name__ == "__main__":
    # 实例化 GoogLeNet（默认 1000 类，启用辅助分类器）
    model = GoogLeNet(num_classes=1000, aux_logits=True)
    
    # 模拟输入：batch_size=2，3 通道，224x224 分辨率
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # 训练模式：输出主预测 + 两个辅助预测
    model.train()
    main_out, aux1_out, aux2_out = model(dummy_input)
    print(f"训练模式输出形状：")
    print(f"主输出: {main_out.shape}")    # 预期: torch.Size([2, 1000])
    print(f"辅助输出1: {aux1_out.shape}")  # 预期: torch.Size([2, 1000])
    print(f"辅助输出2: {aux2_out.shape}")  # 预期: torch.Size([2, 1000])
    
    # 测试模式：仅输出主预测
    model.eval()
    with torch.no_grad():
        test_out = model(dummy_input)
    print(f"\n测试模式输出形状：")
    print(f"主输出: {test_out.shape}")     # 预期: torch.Size([2, 1000])