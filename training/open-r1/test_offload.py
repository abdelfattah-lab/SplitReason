from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np


def get_bigmodel_mask(text, open_tag="<bigmodel>", close_tag="</bigmodel>"):
    mask = [0] * len(text)
    start_index = 0

    while True:
        open_pos = text.find(open_tag, start_index)
        if open_pos == -1:
            break  # no more openings

        close_pos = text.find(close_tag, open_pos + len(open_tag))
        if close_pos == -1:
            # If we can't find a close tag, mark until the end of the text
            for i in range(open_pos, len(text)):
                mask[i] = 1
            break
        else:
            # Mark the region from <bigmodel> ... </bigmodel>
            region_end = close_pos + len(close_tag)
            for i in range(open_pos, region_end):
                mask[i] = 1
            start_index = region_end

    return mask

# model_name = "akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpecReasoner_9k_v2"
# Average coverage: 2.10%          std: 4.73%      min: 0.00%      max: 15.34%
# model_name = "akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpecReasoner"
model_name = "akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpecReasoner_9k_v3"

# Load tokenizer for formatting
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")

# Set up generation pipeline
device = 0 if torch.cuda.is_available() else -1

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # "akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpecReasoner",
    # "akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpecReasoner",
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
offp = []

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
    mask = get_bigmodel_mask(output)
    offload_chars = sum(mask)
    length = len(mask)

    offload_percentage = 100.0 * offload_chars / length
    offp.append(offload_percentage)
    print(f"Question ID: {i}, <bigmodel> occurrences: {count}")

# avg
coverage_list = offp
print(f"Average coverage: {sum(coverage_list) / len(coverage_list):.2f}% \t std: {np.std(coverage_list):.2f}% \t min: {min(coverage_list):.2f}%\t max: {max(coverage_list):.2f}%")

# Print results
print("Occurrences of <bigmodel> per question ID:")
print(bigmodel_counts)

# Save outputs to file as pickle
import pickle
with open("bigmodel_span_text.pkl", "wb") as f: pickle.dump(outputs, f)
# import numpy as np
# from datasets import load_dataset
# mydata = load_dataset("akhauriyash/OpenR1_Math_SpeculativeReasoning")


# offp = []
# for example in mydata['train']['messages']:
#     question = example[1]['content']
#     # get mask
#     mask = get_bigmodel_mask(question)
#     offload_chars = sum(mask)
#     length = len(mask)

#     offload_percentage = 100.0 * offload_chars / length
#     offp.append(offload_percentage)

# # avg
# coverage_list = offp
# print(f"Average coverage: {sum(coverage_list) / len(coverage_list):.2f}% \t std: {np.std(coverage_list):.2f}% \t min: {min(coverage_list):.2f}%\t max: {max(coverage_list):.2f}%")
