import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import struct
import numpy as np


# ──────────────────────────────
# 工具函数
# ──────────────────────────────
def float_to_hex(f: float) -> str:
    """单个 float → 'hXXXXXXXX' 大端 16 进制字符串"""
    return "h{:08x}".format(struct.unpack("!I", struct.pack("!f", f))[0])


def save_array_as_hex(filename: str, array):
    """numpy / tensor 展平成 1-D → 空格分隔 hex 字符串写文件"""
    flat = np.array(array).flatten()
    with open(filename, "w") as f:
        f.write(" ".join(float_to_hex(x) for x in flat))
    print(f"Saved {filename} ({len(flat)} numbers)")


# ──────────────────────────────
# 网络定义
# ──────────────────────────────
class ConvNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5, stride=5)  # [B,1,5,5]
        self.conv2 = nn.Conv2d(1, 1, kernel_size=2)            # [B,1,4,4]
        self.conv3 = nn.Conv2d(1, 10, kernel_size=4)           # [B,10,1,1]

    def forward(self, x):
        x = torch.relu(self.conv1(x))     # [B,1,5,5]
        x = torch.relu(self.conv2(x))     # [B,1,4,4]
        x = self.conv3(x)                 # [B,10,1,1]
        return x.view(x.size(0), -1)      # -> [B,10]

# ──────────────────────────────
# 训练
# ──────────────────────────────
def train_model(num_epochs: int = 3) -> ConvNN:
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="D:/_class_Data/Python/Ventus-OpenCL-Testcase/data",
        train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64, shuffle=True)

    model = ConvNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                done = batch_idx * len(data)
                print(f"Epoch {epoch} [{done}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item():.6f}")
    return model


# ──────────────────────────────
# 评估
# ──────────────────────────────
@torch.no_grad()
def evaluate_model(model: nn.Module):
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root="./data", train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=100, shuffle=False)

    correct = sum((model(data).argmax(1) == target).sum().item()
                  for data, target in test_loader)
    total = len(test_loader.dataset)
    acc = 100.0 * correct / total
    print(f"模型在测试集上的准确率: {acc:.2f}%")
    return acc


# ──────────────────────────────
# 主流程
# ──────────────────────────────
def main():
    model = train_model(num_epochs=10).eval()
    evaluate_model(model)

    # 保存权重/偏置
    save_array_as_hex("conv1_weight.txt", model.conv1.weight.cpu().detach())
    save_array_as_hex("conv1_bias.txt",   model.conv1.bias.cpu().detach())
    save_array_as_hex("conv2_weight.txt", model.conv2.weight.cpu().detach())
    save_array_as_hex("conv2_bias.txt",   model.conv2.bias.cpu().detach())
    save_array_as_hex("conv3_weight.txt", model.conv3.weight.cpu().detach())
    save_array_as_hex("conv3_bias.txt",   model.conv3.bias.cpu().detach())

    # ---- 取 1 个测试样本并逐层推理 ----
    test_dataset = datasets.MNIST(root="./data", train=False, download=True,
                                  transform=transforms.ToTensor())
    test_input, _ = test_dataset[0]               # [1,28,28]
    test_input = test_input.unsqueeze(0)          # -> [1,1,28,28]

    with torch.no_grad():
        out1 = torch.relu(model.conv1(test_input))    # [1,2,24,24]
        out2 = torch.relu(model.conv2(out1))          # [1,1,4,4]
        out3 = model.conv3(out2)                      # [1,10,1,1]
        final_out = out3.view(-1)                     # [10]

    # ---- 保存输入 & 每层输出 ----
    save_array_as_hex("test_input.txt",  test_input.cpu())
    save_array_as_hex("conv1_out.txt",   out1.cpu())
    save_array_as_hex("conv2_out.txt",   out2.cpu())
    save_array_as_hex("conv3_out.txt",   out3.cpu())       # 原尺寸 10×1×1
    save_array_as_hex("test_output.txt", final_out.cpu())  # 展平 10

    print("全部权重、偏置、输入以及各层输出已保存完成。")


if __name__ == "__main__":
    main()
