import re
import os
import collections

# 正则表达式：匹配 RISC-V 指令并过滤注释、标签
ADDR_RE = re.compile(r'/\*.*?\*/')  # 去除注释部分 (/*...*/)

def extract_opcode(line: str) -> str | None:
    """
    从 RISC-V 汇编行提取指令（去除地址和注释）。
    """
    # 跳过空行
    if not line.strip():
        return None

    # 去除地址（如：80000000）
    s = ADDR_RE.sub('', line).strip()
    
    # 如果是注释行或者包含 label，忽略
    if s.startswith(';') or s.startswith('#') or ':' in s:
        return None

    # 只关注包含操作码的行
    tokens = s.split()
    if len(tokens) > 1:  # 确保这一行包含指令
        return tokens[0]  # 返回操作码

    return None


def parse_riscv(path: str) -> collections.Counter[str]:
    """
    解析 RISC-V 汇编文件，统计每条指令的出现次数。
    """
    instruction_counts = collections.Counter()

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 提取操作码（指令）
            opcode = extract_opcode(line)
            if opcode:
                instruction_counts[opcode] += 1

    return instruction_counts


def print_report(instruction_counts: collections.Counter[str]):
    total_instructions = sum(instruction_counts.values())
    print(f"\n=== Total Instruction Count (Total instructions = {total_instructions}) ===")
    for i, (opcode, count) in enumerate(instruction_counts.most_common(), start=1):
        pct = (count / total_instructions * 100.0) if total_instructions else 0.0
        print(f"{i:>3}. {opcode:<24} {count:>10}  ({pct:5.2f}%)")


def main():
    # 输入文件路径
    file_path = input("Enter the path to the RISC-V dump file: ").strip()

    # 检查文件是否存在
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist or is not a valid file.")
        return

    # 解析文件并统计指令
    instruction_counts = parse_riscv(file_path)
    if not instruction_counts:
        print(f"No instructions found. Are you sure this is a valid RISC-V dump file?")
        return

    # 输出统计报告
    print_report(instruction_counts)


if __name__ == '__main__':
    main()
