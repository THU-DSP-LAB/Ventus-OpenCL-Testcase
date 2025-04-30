import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import struct
import numpy as np


# 将 float 转换为 32 位浮点数的 16 进制字符串（带 h 前缀）
def float_to_hex(f):
    # 使用网络字节序（大端）保证格式一致
    return "h{:08x}".format(struct.unpack("!I", struct.pack("!f", f))[0])


# 将 numpy 数组（或任意可转换为一维数组的对象）以空格分隔的 hex 格式保存到文件中
def save_array_as_hex(filename, array):
    # 若 array 为多维数组，则先展平成一维
    flat = np.array(array).flatten()
    hex_strs = [float_to_hex(x) for x in flat]
    with open(filename, "w") as f:
        f.write(" ".join(hex_strs))
    print(f"Saved {filename} with {len(flat)} numbers.")


# 定义简单神经网络：输入 784 → fc1 (128) + ReLU → fc2 (10)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 输入 x 的 shape 为 [batch_size, 1, 28, 28]
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def train_model():
    # 使用 torchvision 下载 MNIST 数据
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    num_epochs = 1  # 为演示仅训练1个 epoch，实际可增加训练轮数
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
    return model


def main():
    # 训练模型
    model = train_model()
    model.eval()

    # 取出各层权重和偏置
    # PyTorch 中 Linear 层的 weight shape 为 [out_features, in_features]，正好与 OpenCL 内核期望的 row-major 存储格式一致
    fc1_weight = model.fc1.weight.detach().cpu().numpy()  # shape: (128, 784)
    fc1_bias = model.fc1.bias.detach().cpu().numpy()  # shape: (128,)
    fc2_weight = model.fc2.weight.detach().cpu().numpy()  # shape: (10, 128)
    fc2_bias = model.fc2.bias.detach().cpu().numpy()  # shape: (10,)

    # 保存权重和偏置到文本文件
    save_array_as_hex("fc1_weight.txt", fc1_weight)
    save_array_as_hex("fc1_bias.txt", fc1_bias)
    save_array_as_hex("fc2_weight.txt", fc2_weight)
    save_array_as_hex("fc2_bias.txt", fc2_bias)

    # 从 MNIST 测试集取一组测试数据，并计算模型输出
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_input, test_target = next(iter(test_loader))  # 取第一个样本
    with torch.no_grad():
        test_output = model(test_input)

    # 将测试输入展平成 784 维向量；输出为 10 维
    test_input_np = test_input.view(-1).detach().cpu().numpy()
    test_output_np = test_output.view(-1).detach().cpu().numpy()

    # 保存测试输入和对应输出
    save_array_as_hex("test_input.txt", test_input_np)
    save_array_as_hex("test_output.txt", test_output_np)

    print("所有数据已保存，可供后续 OpenCL host 代码使用。")


if __name__ == "__main__":
    main()
