# -*- coding: utf-8 -*-
"""
使用项目的 inference.py 进行交互式测试
"""
import os
import sys

# 设置模型路径（使用 modelscope 下载的本地路径）
model_path = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct")

# 构建命令
cmd = f'python inference.py --base_model "{model_path}" --template_name qwen'

print("=" * 60)
print("MedicalGPT 推理测试")
print("=" * 60)
print(f"模型路径: {model_path}")
print("\n正在启动交互式对话...")
print("输入 'exit' 退出，输入 'clear' 清除历史\n")

os.system(cmd)
