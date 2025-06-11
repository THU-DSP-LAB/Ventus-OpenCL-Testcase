import struct
import os

import numpy as np

dec_float = 5.9


def getFP16Str(dec_float=5.9):
    # # 十进制单精度浮点转16位16进制
    hexa = struct.unpack('H', struct.pack('e', dec_float))[0]
    # print()
    hexa = hex(hexa)
    hexa = hexa[2:]
    # print(hexa) # 45e6
    # print(dec_float.tobytes())
    return hexa  # str(dec_float.tobytes())#hexa


def Hex2FP16(hexa: str):
    y = struct.pack("H", int(hexa, 16))
    float = np.frombuffer(y, dtype=np.float16)[0]
    print(float)  # 5.9
    return float


def Hex2FP32(hexa: str):
    y = struct.pack("H", int(hexa, 32))
    float = np.frombuffer(y, dtype=np.float32)[0]
    print(float)  # 5.9
    return float


print("___________result all_____________")
print("truth:", end=" ")
Hex2FP16('c2fd')
print("actual:", end=" ")
Hex2FP16('3c38')
print("___________DP module inside_____________")
print("acc:", end=" ")
Hex2FP16('a2e9')
print("Mul&add:", end=" ")
Hex2FP16('3c46')
print(getFP16Str(-3.494 + 0.0135))
# 1 10000 1011110110
print("___________fp16 demo_____________")
print("Demo:", end=" ")
Hex2FP16('37f3')  # 0 01101 11,1111,0011    03f2: 0011,1111,0011    Actual: 7f3
Hex2FP16('bc0d')  # 1 01111 00,0000,1101    000d: 0000,0000,1101    Actual: 40d
#                                          003357                  203357
print(getFP16Str(0.4968 * -1.013))
Hex2FP16('4711')


def calculate_result(exp_bin, s_bin):
    """
    计算 2^(exp-15) * s，其中 exp 是二进制的指数，s 是二进制的尾数。

    参数:
        exp_bin (str): 表示指数的二进制字符串。
        s_bin (str): 表示尾数的二进制字符串。

    返回:
        float: 计算结果。
    """
    # 将二进制指数转换为整数
    exp = int(exp_bin, 2)

    # 将二进制尾数转换为浮点数
    # 小数点位于最高位和次高位之间
    s = 0.0
    for i, bit in enumerate(s_bin):
        s += int(bit) * (2 ** -(i + 1))

    # 计算结果
    result = (2 ** (exp - 15)) * s
    return result


exp_bin = "01011"  # 示例二进制指数（十进制为 18）
s_bin = "0001000110"  # 示例二进制尾数（十进制为 0.6875）

# 3c46:      0 01111 0001000110
# 0f:          01111
# 04583ff80: 00001000101100000111111111110000000
result = calculate_result(exp_bin, s_bin)
print(f"计算结果: {result}")
Hex2FP16('3c46')

# 000 10011
# 10000000000000000000000000000000000
print(getFP16Str(3.0))  # 4200
print(getFP16Str(-4.0))  # c400

print(getFP16Str(1.0))  # 3c00   0 01111 00,0000,0000
print(getFP16Str(-1.0))  # bc00  1 01111 00,0000,0000
print(getFP16Str(0))  # 0000
print(getFP16Str(-12))  # ca00   1 10010 1000000000
print(getFP16Str(-13))  # ca80   1 10010 1000000000

Hex2FP16('a800')       # a800    1 01010 0000000000
Hex2FP16('cc00')       # cc00    1 10011 0000000000
                                 # 10011 000,0000,1100,0000,...
Hex2FP16('cc0c')       # cc0c    1 10011 0000001100
Hex2FP16('c200')       # cc0c    1 10011 0000001100

# FP16模式下
'''
a: 3c00   0 01111 00,0000,0000
b: bc00   1 01111 00,0000,0000

          1 01111 100 0000 0000 0000 0000 ...
c: bc00   1 01111 00,0000,0000


a: 4200   
b: c400   
          1 10010 110 0000 0000 0000 0000 ...
c: ca00   1 10010 1000000000
'''
