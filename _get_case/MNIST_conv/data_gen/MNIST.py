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


# 定义全卷积神经网络：
# 输入 [batch, 1, 28, 28]
# conv1: 1->16, kernel_size=5, 输出 [batch, 16, 24, 24]
# conv2: 16->32, kernel_size=5, 输出 [batch, 32, 20, 20]
# conv3: 32->10, kernel_size=20, 输出 [batch, 10, 1, 1]，展平后即为 10 维输出
class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)   # output: 16 x 24 x 24
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # output: 32 x 20 x 20
        self.conv3 = nn.Conv2d(32, 10, kernel_size=20) # output: 10 x 1 x 1

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        # 将 [batch, 10, 1, 1] 展平成 [batch, 10]
        x = x.view(x.size(0), -1)
        return x


def train_model():
    # 使用 torchvision 下载 MNIST 数据
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='D:\_class_Data\Python\Ventus-OpenCL-Testcase\data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = ConvNN()
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

    # 取出各层权重和偏置，Conv2d 层的 weight shape 为 [out_channels, in_channels, kernel_height, kernel_width]
    conv1_weight = model.conv1.weight.detach().cpu().numpy()  # shape: (16, 1, 5, 5)
    conv1_bias   = model.conv1.bias.detach().cpu().numpy()    # shape: (16,)
    conv2_weight = model.conv2.weight.detach().cpu().numpy()  # shape: (32, 16, 5, 5)
    conv2_bias   = model.conv2.bias.detach().cpu().numpy()    # shape: (32,)
    conv3_weight = model.conv3.weight.detach().cpu().numpy()  # shape: (10, 32, 20, 20)
    conv3_bias   = model.conv3.bias.detach().cpu().numpy()    # shape: (10,)

    # 保存权重和偏置到文本文件
    save_array_as_hex("conv1_weight.txt", conv1_weight)
    save_array_as_hex("conv1_bias.txt", conv1_bias)
    save_array_as_hex("conv2_weight.txt", conv2_weight)
    save_array_as_hex("conv2_bias.txt", conv2_bias)
    save_array_as_hex("conv3_weight.txt", conv3_weight)
    save_array_as_hex("conv3_bias.txt", conv3_bias)

    # 从 MNIST 测试集取一组测试数据，并计算模型输出
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_input, test_target = next(iter(test_loader))  # 取第一个样本
    with torch.no_grad():
        test_output = model(test_input)

    # 保存测试输入和对应输出
    # 由于模型的输入依然为 [1, 1, 28, 28]，可展平成 784 个数；输出为 10 维
    test_input_np = test_input.view(-1).detach().cpu().numpy()
    test_output_np = test_output.view(-1).detach().cpu().numpy()

    save_array_as_hex("test_input.txt", test_input_np)
    save_array_as_hex("test_output.txt", test_output_np)

    print("所有数据已保存，可供后续 OpenCL host 代码使用。")


if __name__ == "__main__":
    main()
