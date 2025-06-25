#! /usr/bin/python3
import re
from collections import defaultdict
import argparse
import os

# ============== 步骤1：解析RTLSIM日志 ==============
'''
提取所有以sm开头的行。符合以下模式的数据保留，不符合的剔除。注意按照顺序（不论2个模式）添加到1个带排查列表中
模式1：spilt后list_i[]，list_i[3]是指令地址（0x80000070）list_i[4]是指令（0x0302a383），list_i[6]=="x"，list_i[7]是数字，list_i[8]是数据（00000000）
sm   1 warp 0 0x80000070 0x0302a383 x  7  00000000
模式2：spilt后list_i[]，list_i[6]=="v"，list_i[7]是数字，list[8]是mask，list_i[9]~list_i[9+32]是数据
sm   1 warp 0 0x80000290 0x5e02c1d7 v  3 00000000000001110000000000000111 bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a bf59999a
'''

# ------------------------------------------------------------
#  辅助：把十六进制/裸数字统一成 8 位小写十六进制（无 0x 前缀）
# ------------------------------------------------------------
# def norm_hex(h: str) -> str:
#     h = h.lower().strip()
#     if h.startswith("0x"):
#         h = h[2:]
#     return h.zfill(8)  # 保证宽度一致

def parse_rtlsim_log(logfile: str):
    pattern = r'^sm\s+.*'          # 只取以 sm 开头的行（小写）
    entries = []

    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not re.match(pattern, line, re.IGNORECASE):
                continue
            parts = line.strip().split()
            if len(parts) < 9:
                continue    # 数据不完整

            # ---- 通用字段 ----
            # 索引与示例行保持一致：0 sm | 1 1 | 2 warp | 3 0 | 4 addr | 5 instr | 6 x|v | 7 reg | 8+ ...
            # addr   = norm_hex(parts[4])
            # instr  = norm_hex(parts[5])
            addr   = (parts[4])
            instr  = (parts[5])
            regidx = parts[7]
            rtype  = parts[6].lower()
            # print(rtype,regidx,regidx.isdigit())

            # ---- 模式 1：标量寄存器 x ----
            if rtype == "x" and regidx.isdigit() and eval(regidx)!=0:
                # data_field = norm_hex(parts[8])
                data_field = parts[8]
                entries.append({
                    "type": "pattern1",
                    "addr": addr,
                    "instr": instr,
                    # "regidx": regidx,
                    "mask":'',
                    "data": [data_field]
                })

            # ---- 模式 2：向量寄存器 v ----
            elif rtype == "v" and regidx.isdigit():
                if len(parts) >= 9 + 32:
                    # parts[8] 是 32bit mask，忽略；随后 32 个元素为数据
                    # data_fields = [norm_hex(x) for x in parts[9:9+32]]
                    data_fields = [(x) for x in parts[9:9 + 32]]
                    entries.append({
                        "type": "pattern2",
                        "addr": addr,
                        "instr": instr,
                        # "regidx": regidx,
                        "mask": parts[8],
                        "data": data_fields
                    })
    return entries

# ============== 步骤2：日志比对 ==============
# 步骤2：根据获得的待比较信息列表，循环每个条目：
# 在spike_log中找到指令地址（如0x800000d4）对应的spike_log的行，如果对应上，请比较数据是否在spike_log中
# 根据spike log按照两行作为1个条目，匹配到以下三种模式的保留，不符合以下模式的从待比较信息列表中剔除。
'''
模式1：条目第二行split后长度为7，第7个元素且长度为10，以0x开头
第二行地址（0x80000044） 指令（0x00003517） 指令名称（auipc） 后面是 0x开头的10位字符（如0x80003044）。例子如下：
core   0: 0x80000044 (0x00003517) auipc   x10, 0x3
core   0: 3 0x80000044 (0x00003517) x10 0x80003044

core   0: 0x8000000c (0x0d0efed7) vsetvli x29, x29, e32, m1, ta, ma
core   0: 3 0x8000000c (0x0d0efed7) x29 0x00000020 c8_vstart 0x00000000 c3104_vl 0x00000020 c3105_vtype 0x000000d0 c768_mstatus 0x80000600

模式2：条目第二行split后长度为8，第7个元素且长度为10，以0x开头；第八个元素为mem
数据：倒数第三个（0x00000000）和第一个(0x90024034)
core   0: 0x80000074 (0x0342ae03) lw      x28, 52(x5)
core   0: 3 0x80000074 (0x0342ae03) x28 0x00000000 mem 0x90024034
模式3：条目第二行split后长度大于32，提取最后32个元素是数据。例子如下：
数据：0x00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 
core   0: 0x800000d4 (0x5e004057) vmv.v.x v0, x0
core   0: 3 0x800000d4 (0x5e004057) c8_vstart 0x00000000 e32 m1 l32 v0  ffffffff 0x00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 

core   0: 0x800001f4 (0x5e04c157) vmv.v.x v2, x9
core   0: 3 0x800001f4 (0x5e04c157) c8_vstart 0x00000000 e32 m1 l32 v2  77777777 0x00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 


core   0: 0x80000128 (0x0000a0fb) vlw12.v v1, v1, 0
core   0: 3 0x80000128 (0x0000a0fb) e32 m1 l32 v1  00070007 0x00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000000 becccccd be99999a 00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000003 00000000 becccccd be99999a  c8_vstart 0x00000000 mem 0x90000010 mem 0x90000020 mem 0x90000030 mem 0x90000010 mem 0x90000020 mem 0x90000030
core   0: 0x8000042c (0x020fc057) vadd.vx v0, v0, x31
core   0: 3 0x8000042c (0x020fc057) c8_vstart 0x00000000 e32 m1 l32 v0  ffffffff 0x0000000f 0000000e 0000000d 0000000c 0000000b 0000000a 00000009 00000008 00000007 00000006 00000005 00000004 00000003 00000002 00000001 00000000 0000000f 0000000e 0000000d 0000000c 0000000b 0000000a 00000009 00000008 00000007 00000006 00000005 00000004 00000003 00000002 00000001 00000000 

'''


# 匹配模式后，提取每个条目的指令地址（0x800000d4），编码（0x5e004057），名称（vmv.v.x）和数据（ffffffff 0x00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000）

# 比较该条目的信息是否与spike_log的数据信息一致（可能不止1条能在spike中匹配，有匹配spike_log行即代表没问题）
# 如果失败：请输出待比较信息列表的条目信息，以及同指令地址中spike_log信息。
# ============== 步骤2：解析spike日志 ==============
import re

# ---------------- 辅助：把 mask 统一成 32 位二进制 ----------------
def mask_to_bin(mask_str: str) -> str:
    """
    - 若是 8 位十六进制（可带 0x），转 32 位二进制
    - 若本就是 0/1 字符串，则左补 0 保证 32 位
    """
    s = mask_str.lower().replace("0x", "")
    if all(c in "01" for c in s) and len(s) >= 8:        # 已是二进制串
        return s.zfill(32)[-32:]
    else:                                                # 视为十六进制
        return f"{int(s, 16):032b}"
def parse_spike_log(logfilelist):
    """
    仅处理以 "core   " 开头的行；两行一条目（header+data）。
    返回列表 entries，每个元素包含 addr / instr / data。
    """
    entries = []
    buffer  = []

    for logfile in logfilelist:
        with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                # ① 只保留严格前缀 "core   "
                if not raw.startswith("core   "):
                    continue

                buffer.append(raw.strip())

                # ② 两行齐才能解析
                if len(buffer) < 2:
                    continue

                header    = buffer[0].split()
                data_line = buffer[1].split()

                # ---------- 模式 1：标量寄存器 ----------
                if 7 <= len(data_line) <= 32 and data_line[6].startswith("0x"):
                    entries.append({
                        "addr" : data_line[3],
                        "instr": data_line[4][1:-1],
                        "mask":"",
                        "inst": buffer[0],
                        "data" : [data_line[6][2:]]
                    })

                # ---------- 模式 2：mem 写回 ----------
                elif len(data_line) == 8 and data_line[6].startswith("0x") and \
                     data_line[7] == "mem":
                    entries.append({
                        "addr" : data_line[3],
                        "instr": data_line[4][1:-1],
                        "mask": "",
                        "inst": buffer[0],
                        "data" : [data_line[6][2:], data_line[5][2:]]
                    })

                # ---------- 模式 3：向量寄存器 ----------
                elif 50 >len(data_line) > 32:
                    # 直接截取最后 32 个 token，避免正则遗漏
                    vec = [t.lower().replace("0x", "").zfill(8)
                           for t in data_line[-33:]]
                    # print(vec)
                    entries.append({
                        "addr" : data_line[3],
                        "instr": data_line[4][1:-1],
                        "mask": mask_to_bin(vec[-33]),
                        "inst": buffer[0],
                        "data" : vec[-32:]
                    })

                elif any("vlw" in tok for tok in header):
                    # 截取掉从split后的列表，从"c8_vstart"后的全部元素，再取倒数33个元素
                    if "c8_vstart" in data_line:  # 先确认列表里确实有这个标记
                        cut = data_line.index("c8_vstart")  # 找到第一次出现的位置
                        data_line = data_line[:cut]  # 只保留之前的部分
                    vec = [t.lower().replace("0x", "").zfill(8)
                           for t in data_line[-33:]]
                    # print(vec)
                    entries.append({
                        "addr" : data_line[3],
                        "instr": data_line[4][1:-1],
                        "mask": mask_to_bin(vec[-33]),
                        "inst":buffer[0],
                        "data" : vec[-32:]
                    })

                buffer = []

    return entries

# ------------------------------------------------------------
#  步骤 3：日志比对
# ------------------------------------------------------------
# ---------------- 辅助：按 mask 比较两段数据 ----------------
from typing import List  # 放在文件顶部

def masked_equal(data1: List[str], data2: List[str], bin_mask: str) -> bool:
    for i in range(32):
        if bin_mask[i] == "0":
            continue
        if data1[i].lower() != data2[i].lower():
            return False
    return True


def compare_logs(rtlsim_entries, spike_log_list):
    # 解析 spike
    spike_entries = parse_spike_log(spike_log_list)

    # 按地址索引 spike
    addr_map = defaultdict(list)
    for s in spike_entries:
        addr_map[s["addr"]].append(s)

    notmatch = []

    for rtl in rtlsim_entries:
        candidates = addr_map.get(rtl["addr"], [])
        matched = False

        for spk in candidates:
            # ---------- 情况 1：二者都有 mask ----------
            if rtl["mask"] != "" and spk["mask"] != "":
                m_rtl  = mask_to_bin(rtl["mask"])
                m_spk  = mask_to_bin(spk["mask"])
                if m_rtl != m_spk:
                    continue          # mask 本身不一致，直接跳过

                if masked_equal(rtl["data"], spk["data"], m_rtl):
                    matched = True
                    break

            # ---------- 情况 2：无 mask（标量或普通 mem 条目） ----------
            else:
                if rtl["data"] == spk["data"]:
                    matched = True
                    break

        # -------- 记录未匹配 --------
        if not matched:
            notmatch.append(f"{rtl['addr']}_{rtl['instr']}")
            print("\n--------- ❌ 未匹配 ---------")
            print(f"RTLSIM : addr {rtl['addr']} instr {rtl['instr']} mask {rtl.get('mask','--')} "
                  f"data {' '.join(rtl['data'])}")
            for spk in candidates:
                print(f"SPIKE  : addr {spk['addr']} instr {spk['instr']} mask {spk.get('mask','--')} "
                      f"data {' '.join(spk['data'])}")
                print(f"spike inst: {spk['inst']}")

    # -------- 汇总 --------
    print("\n=============== 结果汇总 ===============")
    for item in notmatch:
        print(item)
    print(f"\n总未匹配条目：{len(notmatch)}")

# ------------------------------------------------------------
#  CLI 参数 & Spike 文件收集
# ------------------------------------------------------------
def parse_arguments():
    p = argparse.ArgumentParser("RTLSIM ⟷ SPIKE 日志比对",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-r", "--rtlsim", default="rtlsim.log",
                   help="RTLSIM 日志文件路径")
    p.add_argument("-p", "--spike", default="Fan",
                   help="Spike 日志公共前缀（如 Fan 匹配 Fan*.log）")
    return p.parse_args()


def find_spike_logs(keyword):
    cur = os.getcwd()
    files = [f for f in os.listdir(cur) if keyword in f and f.endswith(".log")]
    if not files:
        raise FileNotFoundError(f"没有找到包含 '{keyword}' 的 Spike 日志")
    files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    return files


# ------------------------------------------------------------
#  main
# ------------------------------------------------------------
if __name__ == "__main__":
    args = parse_arguments()

    # 检查 RTLSIM 日志
    if not os.path.isfile(args.rtlsim):
        print(f"错误：RTLSIM 日志 {args.rtlsim} 不存在")
        exit(1)

    # try:
    spike_logs = find_spike_logs(args.spike)
    rtlsim_entries = parse_rtlsim_log(args.rtlsim)
    # print(spike_logs)
    # print(rtlsim_entries)
    compare_logs(rtlsim_entries, spike_logs)
    # except Exception as e:
    #     print("运行异常：", e)
    #     exit(1)
