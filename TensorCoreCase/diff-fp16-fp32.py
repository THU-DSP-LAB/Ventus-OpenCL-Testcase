import struct
import numpy as np
import os


def getFP16Str(dec_float=5.9):
    # # 十进制单精度浮点转16位16进制
    hexa = struct.unpack('H', struct.pack('e', dec_float))[0]
    # print()
    hexa = hex(hexa)
    hexa = hexa[2:]
    # print(hexa) # 45e6
    # print(dec_float.tobytes())
    return hexa  # str(dec_float.tobytes())#hexa


# def getFP32Str(float_num):
#     # 将浮点数转换为4个字节的二进制数据
#     binary_data = struct.pack('f', float_num)
#     # 将二进制数据转换为十六进制字符串
#     hex_string = binary_data.hex()
#     return hex_string

def getFP32Str(float_value):
    # 将浮点数转换为4字节的二进制格式
    binary_data = struct.pack('!f', float_value)
    # 将二进制数据转换为十六进制字符串
    hex_string = binary_data.hex()
    return hex_string

def Hex2FP16(hexa: str):
    y = struct.pack("H", int(hexa, 16))
    float = np.frombuffer(y, dtype=np.float16)[0]
    # print(float)  # 5.9
    return float


def Hex2FP32(hex_str):
    # 确保输入的16进制字符串长度为8（32位浮点数需要8个十六进制数字）
    if len(hex_str) != 8:
        raise ValueError("Hex string must be exactly 8 characters long")

    # 将16进制字符串转换为字节
    bytes_obj = bytes.fromhex(hex_str)

    # 使用struct.unpack以'>f'格式（大端模式的单精度浮点数）来解析字节
    float_num = struct.unpack('>f', bytes_obj)[0]

    return float_num
def hex_to_float(h):
    i = int(h,16)
    return struct.unpack('<f',struct.pack('<I', i))[0]

if __name__ == '__main__':
    # 假设文件名为 'data.txt'
    file_path = 'data.txt'

    # 创建一个空的大列表用于存储所有行的列表
    big_list = []

    # 打开文件并逐行读取
    with open(file_path, 'r') as file:
        for line in file:
            # 去除每行末尾的换行符并按空格分割字符串
            parts = line.split()
            # 将分割后的字符串放入列表中
            line_list = [parts[0], parts[1]]
            # 将列表添加到大列表中
            big_list.append(line_list)

    # 打印结果查看
    for sublist in big_list:
        print(sublist)
    # for()
    val = []
    for i in big_list:
        temp = hex_to_float('0x'+i[0])-hex_to_float('0x'+i[1])
        val.append(temp)
        print(hex_to_float('0x'+i[0]),hex_to_float('0x'+i[1]),temp)

    print(np.mean(val),np.std(val))

    print("++++++++++++++++++++++++++++")

    # print(hex_to_float('0x'+'BE10A000'))
    # print(Hex2FP16("b085"), Hex2FP32("be000085"))
    # print(Hex2FP16("b085"), Hex2FP32("BE10A000"))
    # print(Hex2FP16("3eb6"), Hex2FP32("3f8002b6"))
    # print(Hex2FP16("3eb6"), Hex2FP32("3FD6C000"))
    # print(Hex2FP32("52e4d6bf"))
    #
    #
    # print(getFP32Str(9.54),Hex2FP32("4118a3d7"))
    #
    # print(Hex2FP32("bf4ef13e"))#-0.8

    # l = ["3e311cdd","40ec08e7"]
    # l2 = ["00e68000","3e311cdd"]
    # def get_p(l:list):
    #     print(Hex2FP32(l[0]),Hex2FP32(l[1]))
    #
    # # 7.20190716e+00
    #
    # get_p(l)
    # get_p(l2)
    # print(Hex2FP16("4734"),Hex2FP32("00e68000"))



