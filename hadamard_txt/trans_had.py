import torch
import numpy as np


def ascii_to_signed_tensor_string(filepath):
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 仅保留 + 和 -
            filtered = [c for c in line if c in ['+', '-']]
            if not filtered:
                continue
            # 转换为 +1/-1 表示字符串
            row_str = ', '.join(['+1' if c == '+' else '-1' for c in filtered])
            rows.append(f'        [{row_str}]')

    tensor_str = 'torch.FloatTensor([\n' + ',\n'.join(rows) + '\n    ])'
    return tensor_str


# 路径为你上传的文件
filepath = '/home/shaoyuantian/program/MXFP4-LLM/hadamard_txt/had.148.will.txt'
tensor_formatted = ascii_to_signed_tensor_string(filepath)

# 打印前几行看看效果
print(tensor_formatted[:500])  # 可根据需要打印全量或保存

# 也可以保存为 .py 文件以便直接导入使用
with open("had_tensor.py", "w") as f:
    f.write("import torch\n\n")
    f.write("H = " + tensor_formatted + "\n")
print("✅ 转换完成，保存为 had_96_tensor.py")
