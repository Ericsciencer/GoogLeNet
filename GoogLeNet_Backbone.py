import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ----------------------
# 1. GoogLeNet 模型定义（核心保留原文架构，针对CIFAR-10做微小适配）
# ----------------------
class InceptionModule(nn.Module):
    """
    核心 Inception 模块（和原论文完全一致，无修改）
    包含4个并行分支，多尺度特征提取后在通道维度拼接
    """
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        # 分支1: 1x1卷积
        self.branch1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        # 分支2: 1x1降维 + 3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 分支3: 1x1降维 + 5x5卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        # 分支4: 3x3最大池化 + 1x1卷积
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
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class AuxiliaryClassifier(nn.Module):
    """
    辅助分类器（针对CIFAR-10的修改：调整全连接层输入维度）
    原论文输入是128*4*4，CIFAR-10下特征图更小，改为128*2*2
    """
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        # 修改点：适配CIFAR-10的小特征图
        self.fc1 = nn.Linear(128 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet_CIFAR10(nn.Module):
    """
    完整 GoogLeNet（针对CIFAR-10的修改：调整前几层卷积/池化）
    1. 原论文第一层7x7 stride=2 -> 修改为3x3 stride=1（避免32x32图像过早缩小）
    2. 移除原论文第一个池化层（同样为了保留空间尺寸）
    3. 核心Inception模块与辅助分类器逻辑完全保留原论文
    """
    def __init__(self, num_classes=10, aux_logits=True):
        super(GoogLeNet_CIFAR10, self).__init__()
        self.aux_logits = aux_logits

        # ------------------------------
        # 修改点1：适配32x32输入的前几层
        # ------------------------------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # 原论文7x7 stride=2
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 修改点2：移除过早池化
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ------------------------------
        # Inception模块（与原论文完全一致）
        # ------------------------------
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

        # 辅助分类器
        if self.aux_logits:
            self.aux1 = AuxiliaryClassifier(512, num_classes)
            self.aux2 = AuxiliaryClassifier(528, num_classes)

        # 最终分类层（与原论文一致：全局平均池化+无全连接层）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 前几层（修改后）
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)

        # Inception 3a-3b
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Inception 4a-4e + 辅助分类器
        x = self.inception4a(x)
        aux1_out = self.aux1(x) if self.training and self.aux_logits else None
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2_out = self.aux2(x) if self.training and self.aux_logits else None
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

        if self.training and self.aux_logits:
            return x, aux1_out, aux2_out
        return x


# ----------------------
# 2. 数据加载与预处理
# ----------------------
def get_data_loaders(batch_size=64):
    # CIFAR-10的均值和标准差（标准归一化参数）
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    # 训练集：简单数据增强（随机水平翻转）
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    # 测试集：仅归一化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    # 加载数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ----------------------
# 3. 训练函数（核心修改：处理辅助分类器损失，计算训练集准确率）
# ----------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0  # 累计训练集预测正确的样本数
    total = 0    # 累计训练集总样本数

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # 前向传播：现在只获取主输出
        outputs = model(images)
        # 只计算主损失
        loss = criterion(outputs, labels)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 计算当前batch的训练准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 累计总损失
        total_loss += loss.item() * images.size(0)
    
    # 计算当前epoch的平均损失和平均准确率
    avg_train_loss = total_loss / len(train_loader.dataset)
    avg_train_acc = correct / total
    return avg_train_loss, avg_train_acc


# ----------------------
# 4. 测试函数
# ----------------------
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 测试时不使用辅助分类器
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# ----------------------
# 5. 主程序
# ----------------------
if __name__ == '__main__':
    # 基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    lr = 0.005  
    num_epochs = 20  

    # 初始化核心组件
    model = GoogLeNet_CIFAR10(num_classes=10, aux_logits=False).to(device)
    
    # Kaiming 初始化（专门针对 ReLU）
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(weights_init)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    train_loader, test_loader = get_data_loaders(batch_size)

    # 初始化列表，存储每个epoch的3个核心指标
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 训练循环
    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        # 接收训练损失+训练准确率
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_acc = test(model, test_loader, device)
        
        # 把当前epoch的指标存入列表，用于后续绘图
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        # 打印训练日志
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型权重
    torch.save(model.state_dict(), 'googlenet_cifar10.pth')
    print("Model saved as googlenet_cifar10.pth")

    # ----------------------
    # 可视化绘图
    # ----------------------
    epochs = range(1, num_epochs + 1)

    # 设置画布大小
    plt.figure(figsize=(10, 7))

    # 绘制3条曲线
    plt.plot(epochs, train_loss_list, 'b-', linewidth=2, label='train loss')
    plt.plot(epochs, train_acc_list, 'm--', linewidth=2, label='train acc')
    plt.plot(epochs, test_acc_list, 'g--', linewidth=2, label='test acc')

    # 图表配置
    plt.xlabel('epoch', fontsize=18)
    plt.xticks(range(2, num_epochs+1, 2))
    plt.ylim(0, 2.4)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=18)
    plt.title('GoogLeNet Training Metrics (CIFAR-10)', fontsize=16)

    plt.savefig('googlenet_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()