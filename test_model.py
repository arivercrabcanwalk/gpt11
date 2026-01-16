# -*- coding: utf-8 -*-
"""
简单测试脚本 - 使用 modelscope 下载模型并测试推理效果
"""
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 使用 Qwen2.5-0.5B-Instruct 小模型测试
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"正在加载模型: {model_name}")
print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

print("模型加载完成！\n")

# 测试医疗问答
test_questions = [
    "小孩发烧怎么办？",
    "感冒和流感有什么区别？",
    "高血压患者饮食需要注意什么？"
]

for question in test_questions:
    print(f"问题: {question}")
    
    messages = [
        {"role": "system", "content": "你是一个专业的医疗助手，请用简洁专业的语言回答问题。"},
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"回答: {response}\n")
    print("-" * 50)
