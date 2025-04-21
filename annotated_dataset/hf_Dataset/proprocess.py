from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from huggingface_hub import login
import os

our_dataset = load_from_disk("OpenR1_Math_SpeculativeReasoning")

print("Length pre filtering:", len(our_dataset))
# First, filter out the examples you want to exclude
filtered_dataset = our_dataset.filter(
    lambda example: (
        example["annotated_generations"] and 
        "Error processing Due To" not in example["annotated_generations"][0] and 
        len(example["annotated_generations"][0]) > 1024
    )
)

min_len = min(
    len(example["annotated_generations"][0])
    for example in filtered_dataset
    if example["annotated_generations"]
)
shortest_examples = our_dataset.filter(
    lambda example: (
        example["annotated_generations"] and 
        len(example["annotated_generations"][0]) == min_len
    )
)

print("Example of shortest example:", shortest_examples['generations'][0][0])
print("Example of shortest example:", shortest_examples['annotated_generations'][0][0])

print("Length post filtering:", len(filtered_dataset))
def replace_with_annotated(example):
    if example["annotated_generations"]:
        example["messages"][1]["content"] = example["annotated_generations"][0].replace("<\\bigmodel>", "\n </bigmodel> \n")
        example["messages"][1]["content"] = example["annotated_generations"][0].replace("<bigmodel>", "\n <bigmodel> \n")
        
    return example

updated_dataset = filtered_dataset.map(replace_with_annotated)

updated_dataset.push_to_hub("akhauriyash/OpenR1_Math_SpeculativeReasoning")