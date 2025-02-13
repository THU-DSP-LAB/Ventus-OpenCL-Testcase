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


print(Hex2FP16('00f0'))
print(Hex2FP16('b49a'))
# FP16 0108 1.574e-05
# FP32 37840000 0.000015735626220703125

# print(Hex2FP32('37700000'))
