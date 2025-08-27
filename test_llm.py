from llama_cpp import Llama

# 确保这里的路径和你的 main.py 中完全一致
MODEL_PATH = "./models/qwen1_5-1_8b-chat-q4_k_m.gguf"

print("--- 正在加载模型... ---")
# 直接使用 llama-cpp-python 的 Llama 类来加载模型
llm = Llama(
  model_path=MODEL_PATH,
  n_ctx=3900,      # 上下文窗口大小
  n_threads=4,     # 树莓派4B有4个核心
  n_gpu_layers=0   # 确认使用CPU
)
print("--- 模型加载成功！ ---")

# 这是Qwen1.5模型的标准对话格式
prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请用中文介绍一下你自己。<|im_end|>
<|im_start|>assistant
"""

print("\n--- 正在生成回答... ---")
print(f"--- 发送给模型的Prompt: ---\n{prompt}")

# 直接调用模型进行补全
output = llm(
  prompt,
  max_tokens=256, # 你希望生成的最大token数
  stop=["<|im_end|>"], # 在遇到这个标记时停止生成
  echo=False # 不回显输入的prompt
)

print("\n--- 模型返回的原始Output: ---")
print(output)

print("\n--- 提取并打印最终回答: ---")
if output and 'choices' in output and len(output['choices']) > 0:
    response_text = output['choices'][0]['text']
    print(response_text)
else:
    print("【错误】模型返回了空的或无效的结构。")
