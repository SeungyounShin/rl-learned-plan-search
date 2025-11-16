from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Seungyoun/Qwen3-8B-Non-Thinking"
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False , add_generation_prompt=True)
print(prompt)