# -*- coding: utf-8 -*-
"""
交互式医疗问答测试脚本
"""
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("MedicalGPT 交互式医疗问答测试")
print("=" * 60)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"正在加载模型: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)
model.eval()

print("模型加载完成！")
print("-" * 60)
print("输入 'exit' 退出，输入 'clear' 清除对话历史")
print("-" * 60)

history = []

while True:
    try:
        user_input = input("\n你: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见！")
        break
    
    if not user_input:
        continue
    
    if user_input.lower() == 'exit':
        print("再见！")
        break
    
    if user_input.lower() == 'clear':
        history = []
        print("对话历史已清除。")
        continue
    
    # 构建消息
    messages = [
        {"role": "system", "content": "你是一个专业的医疗助手，请用简洁专业的语言回答问题。注意：你的回答仅供参考，不能替代专业医生的诊断。"}
    ]
    
    # 添加历史对话
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    
    messages.append({"role": "user", "content": user_input})
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print("\n医疗助手: ", end="", flush=True)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    
    # 保存到历史
    history.append([user_input, response])
    
    # 只保留最近5轮对话
    if len(history) > 5:
        history = history[-5:]
