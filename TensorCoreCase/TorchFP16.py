import torch
import numpy as np


def Mat_Mul(m: int, n: int, k: int):
    # 确保你的矩阵是FP16类型
    # A = torch.randn(m, k, dtype=torch.float16)
    # B = torch.randn(k, n, dtype=torch.float16)
    # C = torch.randn(m, n, dtype=torch.float16)

    A = torch.tensor(np.loadtxt('testData/a1.txt'), dtype=torch.float16)
    B = torch.tensor(np.loadtxt('testData/b1.txt'), dtype=torch.float16)
    C = torch.tensor(np.loadtxt('testData/c1.txt'), dtype=torch.float16)
    print(A,B,C)
    # 执行FP16矩阵加法
    D = torch.matmul(A, B) + C


    # 打印结果
    print("FP16矩阵乘加结果：", D)
    np.savetxt('testData/d1_torch.txt', D.numpy())


Mat_Mul(16, 8, 8)
