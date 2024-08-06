import struct
import numpy as np
import csv


def getFP16Str(dec_float=5.9):
    # 十进制单精度浮点转16位16进制
    hexa = struct.unpack('H', struct.pack('e', dec_float))[0]
    print(hexa)
    hexa = hex(hexa)
    hexa = hexa[2:]
    # print(hexa) # 45e6
    return hexa


def Hex2FP16(hexa: str):
    y = struct.pack("H", int(hexa, 16))
    float = np.frombuffer(y, dtype=np.float16)[0]
    # print(float)  # 5.9
    return float


def get_diff(std,out):
    cnt = 0
    diff = []
    for i in range(4):
        # print(Hex2FP16(std[i*4:(i+1)*4]),Hex2FP16(out[i*4:(i+1)*4])))
        if Hex2FP16(std[i * 4:(i + 1) * 4]) == Hex2FP16(out[i * 4:(i + 1) * 4]):
            cnt+=1
            diff.append(0)
        else:
            diff.append(Hex2FP16(std[i * 4:(i + 1) * 4]) - Hex2FP16(out[i * 4:(i + 1) * 4]))
    return cnt,diff

def get_diff_all(std,out):
    diff = []
    cnt = 0
    for i in range(len(std)):
        cnt_i,diff_i = get_diff(std[i], out[i])
        cnt+=cnt_i
        diff+=diff_i
    print("success",cnt/128)
    print(np.mean(diff),np.std(diff))
    return cnt,np.mean(diff),np.std(diff)





if __name__ == '__main__':
    # std = 'c65a3a442a70c556'
    # out = "c65c3a442a60c557"
    # 打开CSV文件
    with open('results.csv', 'r', encoding='UTF-8') as csvfile:
        # 创建CSV阅读器
        datareader = csv.reader(csvfile)

        # 初始化一个字典，将每一列的值存储在列表中
        numpy_l = []
        torch_l = []
        cuda_l = []
        res = []

        # 遍历CSV文件的每一行
        for row in datareader:
            # for i, value in enumerate(row):
                # column_lists[i].append(value)  # 将值添加到对应的列列表中
            # print(row)
            # res_t = row.split(',')
            res_t = row
            numpy_l.append(res_t[0].replace('\uFEFF', ''))
            torch_l.append(res_t[1])
            cuda_l.append(res_t[2])
            res.append(res_t[3])


    # # 输出每一列的列表
    # for i, column_list in column_lists.items():
    #     print(f"Column {i}: {column_list}")
    #
    # print(column_list[0])# = '3563c04fc54c43a0'
    #

    # print(numpy_l)
    # print(torch_l)
    # print(cuda_l)
    # print(res)

    get_diff_all(numpy_l,res)
    get_diff_all(torch_l, res)
    get_diff_all(cuda_l, res)

    print("___________________")
    get_diff_all(numpy_l,torch_l)
    get_diff_all(torch_l, cuda_l)
    get_diff_all(cuda_l, numpy_l)
    # getFP16Str()