import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import neuron, functional, surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F
from manual_model import spiking_resnet18

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),          # 转换为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 加载 SVHN 数据集
train_dataset = datasets.SVHN(root='../data', split='train', download=True, transform=transform)
test_dataset = datasets.SVHN(root='../data', split='test', download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=4)


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # 重置脉冲神经元的状态
        functional.reset_net(model)
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 重置脉冲神经元的状态
            functional.reset_net(model)

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

torch.cuda.set_device(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化 Spiking ResNet-18 并移动到 GPU（如果有的话）
model = spiking_resnet18(num_classes=10, T=1).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, criterion, optimizer, epoch)
    test(model, device, test_loader, criterion)

torch.save(model.state_dict(), 'spiking_resnet18_svhn.pth')
