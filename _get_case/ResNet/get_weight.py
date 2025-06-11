import torch
import torchvision.models as models
import os
import numpy as np

# 加载模型
model = models.resnet18(pretrained=True)
state_dict = model.state_dict()

# 创建保存目录
os.makedirs("weights_txt", exist_ok=True)

# 保存函数：扁平化张量并写入 txt
def save_txt(tensor: torch.Tensor, filename: str):
    array = tensor.cpu().numpy().flatten().astype(np.float32)
    with open(os.path.join("weights_txt", filename + ".txt"), "w") as f:
        for v in array:
            f.write(f"{v:.6f}\n")

# 遍历所有参数，全部保存
for name, tensor in state_dict.items():
    name_txt = name.replace(".", "_")  # PyTorch 名称含 .，不便作为文件名
    save_txt(tensor, name_txt)

print("✅ 所有权重已保存到 weights_txt/*.txt")
