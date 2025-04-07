import os
import re
import difflib
import openai
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
# load a tokenizer meta-llama/Llama-3.2-3B
from transformers import AutoTokenizer
import concurrent.futures
from tqdm import tqdm
from pprint import pprint
import re
import difflib
import openai
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoTokenizer
import os
import time
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import spacy
import httpx, json, time, os
import re
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

load_dotenv("./../.env")
current_model = "deepseek-chat"
START_BIGMODEL = "<bigmodel>"
END_BIGMODEL = "<\\bigmodel>"

if current_model in ["deepseek-reasoner", "deepseek-chat"]:
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_deepseek_r1_for_difficult_snippets_fraction(
    cot_text,
    bidx,
    client,
    current_model,
    desired_fraction=0.2  # e.g., 20%
):
    """
    Example using DeepSeek R1 model (OpenAI-compatible) to return the
    'most difficult' or 'hardest' snippets from the chain-of-thought,
    totaling about `desired_fraction` (10-20%) of the entire text.

    - We ask the model to produce short 'snippets' (verbatim).
    - Each snippet is wrapped in:

        -----SNIPPET START-----
        <verbatim content>
        -----SNIPPET END-----

    - Then we parse them via a regex.
    - If the model doesn't comply perfectly, we fallback to a naive line-based approach.

    Returns:
      A list of snippet strings, each containing the verbatim text from the CoT.
    """
    total_chars = len(cot_text)
    target_chars = int(total_chars * desired_fraction)
    system_message = f"""
You are DeepSeek-R1, a large reasoning model.
We will provide you with a chain-of-thought (CoT).

Your task:
1. Identify the logically complex or difficult portions of the CoT.
2. Extract them verbatim in short snippets (a few lines, an equation, etc.)—no huge paragraphs.
3. The total length of the extracted snippets combined should be about {int(desired_fraction * 100)}%
of the entire CoT (roughly {target_chars} characters), focusing only on the hardest parts.
4. Return each snippet enclosed in:

-----SNIPPET START-----
<verbatim text>
-----SNIPPET END-----

5. No extra commentary or text before/between/after these tags. 
6. If fewer parts truly deserve mention, return fewer.
7. Do not paraphrase; preserve exact whitespace and punctuation from the CoT.
8. If the problem is too easy, try to atleast give some 'relatively difficult' snippets regardless.
9. Try to atleast provide 3-4 snippets even if the CoT is trivial.
""".strip()

    user_message = f"""\
Chain-of-thought (CoT):
{cot_text}

Please produce short difficult/complex snippets, totalling ~{int(desired_fraction * 100)}%
of the CoT length (about {target_chars} characters if possible).
Each snippet must be verbatim, enclosed in the tags:

-----SNIPPET START-----
<verbatim text>
-----SNIPPET END-----

Provide some snippets even if the CoT is trivial. No other text or commentary outside the tags.
""".strip()

    templist = [0, 0.3, 0.6]
    max_attempts = 3
    for attempt in range(max_attempts):
        temperature = templist[attempt]
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
            )
            break
        except Exception as e:
            # write system_message and user_message to a file
            # generate unique id
            uid = f"{bidx}_{attempt}"
            if not os.path.exists("attempts"):
                os.makedirs("attempts")
            with open(f"attempts/system_message+{uid}.txt", "a") as f:
                f.write(f"System message: {system_message}\n")
            with open(f"attempts/user_message+{uid}.txt", "a") as f:
                f.write(f"User message: {user_message}\n")
            with open(f"attempts/response+{uid}.txt", "a") as f:
                f.write(f"Error: {e}\n")
            print(f"\n\n\n Loc-Index: {bidx}\t[Attempt {attempt+1}/{max_attempts}]\tResponse error: {e}\n\n\n")
            # Otherwise, wait for 60 seconds and retry
            if attempt < max_attempts - 1:
                time.sleep(60)
    raw_output = response.choices[0].message.content

    # 4) Regex to capture the snippet blocks
    pattern = (
        r"-{3,}\s*SNIPPET\s+START\s*-{3,}\s*"
        r"(.*?)"
        r"\s*-{3,}\s*SNIPPET\s+END\s*-{3,}"
    )
    matches = re.findall(pattern, raw_output, flags=re.DOTALL)

    # 5) Fallback if the model did not use the tags
    if not matches:
        # fallback: naive approach (just grab lines from the raw output)
        lines = [ln.strip('\r') for ln in raw_output.split('\n') if ln.strip()]
        return lines  # or maybe lines[:some_number] if you like

    # 6) Return the snippets exactly as captured
    snippets = [m for m in matches]

    return snippets


def find_best_substring_match_simple(
    snippet, 
    text, 
    min_ratio=0.5, 
    refine_threshold=0.95, 
    boundary_search_size=50,
    boundary_sub_len=5
):
    snippet_len = len(snippet)
    text_len = len(text)
    if snippet_len > text_len:
        return None
    snippet_lower = snippet.lower()
    text_lower = text.lower()
    
    best_ratio = 0.0
    best_start = None
    best_end = None
    
    for i in range(text_len - snippet_len + 1):
        candidate = text_lower[i : i + snippet_len]
        ratio = difflib.SequenceMatcher(None, snippet_lower, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
            best_end = i + snippet_len
    if best_ratio < min_ratio:
        return None
    
    if best_ratio < refine_threshold:
        start_sub = snippet_lower[:boundary_sub_len]
        end_sub   = snippet_lower[-boundary_sub_len:]
        if start_sub:
            left_bound = max(0, best_start - boundary_search_size)
            right_bound = min(text_len, best_start + boundary_search_size)
            offset_region = text_lower[left_bound : right_bound]
            local_idx = offset_region.find(start_sub)
            if local_idx != -1:
                new_start = left_bound + local_idx
                if abs(new_start - best_start) <= boundary_search_size:
                    best_start = new_start
        if end_sub:
            left_bound = max(0, best_end - boundary_search_size - boundary_sub_len)
            right_bound = min(text_len, best_end + boundary_search_size)
            offset_region = text_lower[left_bound : right_bound]
            local_idx = offset_region.find(end_sub)
            if local_idx != -1:
                new_end = left_bound + local_idx + boundary_sub_len
                if abs(new_end - best_end) <= boundary_search_size:
                    best_end = new_end
    return (best_start, best_end, best_ratio)



def insert_annotations(gen_text, matches):
    annotated = []
    last_pos = 0

    for (start_idx, end_idx) in matches:
        annotated.append(gen_text[last_pos:start_idx])
        annotated.append(f"{START_BIGMODEL}{gen_text[start_idx:end_idx]}{END_BIGMODEL}")
        last_pos = end_idx
    annotated.append(gen_text[last_pos:])
    return "".join(annotated)



def process_example(example, bidx=None, top_percent=0.4, llm_top_k=None):
    new_annotated = []
    
    for genid, gen_text in enumerate(example["generations"]):
        if genid > 0:
            # Stop annotated anything except the first example
            new_annotated.append(gen_text)
        try:
            hard_candidates = call_deepseek_r1_for_difficult_snippets_fraction(gen_text, bidx, client, current_model, desired_fraction=0.2)
            match_offsets = []
            for hc in hard_candidates:
                best = find_best_substring_match_simple(hc, gen_text, min_ratio=0.5)
                if best is not None:
                    (start_idx, end_idx, ratio) = best
                    match_offsets.append((start_idx, end_idx))
            match_offsets.sort(key=lambda x: x[0])

            print("Match intervals:\t", match_offsets)
            merged = []
            for (s, e) in match_offsets:
                if not merged:
                    merged.append((s, e))
                else:
                    old_s, old_e = merged[-1]
                    if s > old_e + 10:
                        merged.append((s, e))
                    else:
                        merged[-1] = (old_s, max(old_e, e))
            print("Merged intervals:\t", merged)
            offload_chars = sum([x[1] - x[0] for x in merged])
            gen_text_tokens = len(tokenizer.tokenize(gen_text))
            offload_tokenset = [len(tokenizer.tokenize(gen_text[x[0]:x[1]])) for x in merged]
            offload_tokens = sum(offload_tokenset)

            total_chars = len(gen_text)
            offloading_characteristic = f"{bidx}:\tOffloaded {offload_chars} / {total_chars} total chars\t Tokens: {offload_tokens} / {gen_text_tokens} \t\t Per-offload #Tokens: {offload_tokens / len(merged)}, \t\t Percentage Offload: {offload_chars / total_chars:.2%}"
            print(offloading_characteristic)
            ex_uuid = example["uuid"]
            if not os.path.exists("offloading_characteristic.csv"):
                with open("offloading_characteristic.csv", "w") as f:
                    f.write("uuid,genid,offload_chars,total_chars,offload_tokens,gen_text_tokens,offload_tokens_per_sentence,percentage_offload\n")
            with open("offloading_characteristic.csv", "a") as f:
                f.write(f"{ex_uuid},{genid},{offload_chars},{total_chars},{offload_tokens},{gen_text_tokens},{offload_tokens / len(merged)},{offload_chars / total_chars:.2%}\n")
            with open("merged_intervals.csv", "a") as f:
                f.write(f"{ex_uuid},{merged}\n")
            annotated_text = insert_annotations(gen_text, merged)
            new_annotated.append(annotated_text)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"LocIDX {bidx} Failed: Error processing example\n{e}")
            new_annotated.append(f"{START_BIGMODEL}Error processing Due To {e} \n {END_BIGMODEL}")

    example["annotated_generations"] = new_annotated
    return example




def process_batch_in_parallel(batch, top_percent=0.4, llm_top_k=None):
    annotated_batch = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
        futures = []
        for bidx, ex in enumerate(batch):
            futures.append(executor.submit(process_example, ex, bidx, top_percent, llm_top_k))
        
        for f in concurrent.futures.as_completed(futures):
            annotated_example = f.result()
            annotated_batch.append(annotated_example)
    
    return annotated_batch

ds = load_dataset("open-r1/OpenR1-Math-220k", "default")

ds_small = ds["train"].select(range(15000))

processed_indices = set()
if os.path.exists("processed_ids.txt"):
    with open("processed_ids.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                processed_indices.add(int(line))

if os.path.exists("OpenR1_Math_Annotated_DeepSeek"):
    partial_dataset = load_from_disk("OpenR1_Math_Annotated_DeepSeek")
    genlens = [(idx, [len(t) for t in annots]) 
               for idx, annots in enumerate(partial_dataset["annotated_generations"])]
    to_remove = []
    for idx, lens in genlens:
        if any(length < 160 for length in lens):
            print(f"Discarding example {idx} with lengths {lens}")
            processed_indices.discard(idx)
            to_remove.append(idx)
    keep_indices = [i for i in range(len(partial_dataset)) if i not in to_remove]
    partial_dataset = partial_dataset.select(keep_indices)
    all_results = partial_dataset.to_list()
else:
    partial_dataset = None
    all_results = []

num_parallel = 50
chunk_counter = 0

for start_idx in tqdm(range(0, len(ds_small), num_parallel), desc="Processing batches"):
    end_idx = min(start_idx + num_parallel, len(ds_small))
    batch_indices = range(start_idx, end_idx)
    
    # Build a sub-batch that only includes unprocessed examples
    sub_batch = []
    sub_batch_indices = []
    for i in batch_indices:
        if i not in processed_indices:
            sub_batch.append(ds_small[i])
            sub_batch_indices.append(i)
    
    if not sub_batch:
        continue  # skip if entire chunk was already processed

    # Now process in parallel
    annotated_sub_batch = process_batch_in_parallel(sub_batch)
    # exit(0)
    # Append results to `all_results`, also mark them in processed file
    for ex_idx, annotated_example in zip(sub_batch_indices, annotated_sub_batch):
        all_results.append(annotated_example)
        # Mark as processed
        with open("processed_ids.txt", "a") as f:
            f.write(str(ex_idx) + "\n")
        processed_indices.add(ex_idx)

    chunk_counter += 1
    # Maybe save partial checkpoint every chunk
    if chunk_counter % 1 == 0:
        partial_dataset = Dataset.from_list(all_results)
        if current_model in ["deepseek-reasoner", "deepseek-chat"]:
            partial_dataset.save_to_disk("OpenR1_Math_Annotated_DeepSeek")
        else:
            partial_dataset.save_to_disk("OpenR1_Math_Annotated_GPT")

        print(f"Checkpoint saved at chunk {chunk_counter}")

# Optionally, after the loop finishes, save the final dataset:
final_dataset = Dataset.from_list(all_results)
if current_model in ["deepseek-reasoner", "deepseek-chat"]:
    final_dataset.save_to_disk("OpenR1_Math_Annotated_DeepSeek_Final")
else:
    final_dataset.save_to_disk("OpenR1_Math_Annotated_GPT_Final")

print("Done!")
