from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm

# Load tokenizer for formatting
tokenizer = AutoTokenizer.from_pretrained("akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpecReasoner")

# Load dataset
dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")

# Set up generation pipeline
device = 0 if torch.cuda.is_available() else -1

model = AutoModelForCausalLM.from_pretrained(
    # "akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpecReasoner",
    "akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpecReasoner",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16)

# Helper to format messages using chat template
def format_chat_prompt(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Dictionary to store <bigmodel> counts
bigmodel_counts = {}
outputs = {}

# Loop over first 10 questions
for i in tqdm(range(10)):
    question = dataset[i]["problem"]

    # Format the chat prompt properly
    messages = [{"role": "user", "content": question}]
    prompt = format_chat_prompt(messages)

    # Generate response
    output = generator(prompt, max_new_tokens=4096, return_full_text=False, do_sample=True, temperature=0.7)[0]["generated_text"]
    outputs[i] = output

    # Count <bigmodel> occurrences
    # count = output.count("<bigmodel>")
    count = output.count("bigmodel")
    if count > 0:
        bigmodel_counts[i] = count
    print(f"Question ID: {i}, <bigmodel> occurrences: {count}")

# Print results
print("Occurrences of <bigmodel> per question ID:")
print(bigmodel_counts)

# Save outputs to file as pickle
import pickle
with open("bigmodel_span_text.pkl", "wb") as f: pickle.dump(outputs, f)
