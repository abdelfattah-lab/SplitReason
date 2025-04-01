import os
import re
import difflib
import openai
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
# load a tokenizer meta-llama/Llama-3.2-3B
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

load_dotenv("./../.env")
current_model = "gpt-4o"
START_BIGMODEL = "<bigmodel>"
END_BIGMODEL = "<\\bigmodel>"

if current_model == "deepseek-reasoner":
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

##################################
# 1. Finding (start, end) spans
##################################

def find_sentence_spans(text):
    """
    Find "sentence" boundaries in `text` where
      - '.' is followed by a space, newline, or tab, OR
      - '!' or '?' is followed by space, newline, or tab,
      - plus we also consider '.' '!' or '?' at the very end of the text.
    Returns a list of (start, end) spans, each of which is a substring that
    we will consider one "sentence".
    
    By returning spans rather than splitting & rejoining, we ensure that
    we can reconstruct the original text *exactly*.
    """
    pattern = re.compile(
        r'(?:\.(?=[ \n\t])|\?(?=[ \n\t])|!(?=[ \n\t])'  # e.g. ". ", ".\n", "?\t", etc.
        r'|\.$|!$|\?$)'                               # or if punctuation is at the very end of the text
    )
    
    spans = []
    last_start = 0
    
    for match in pattern.finditer(text):
        # `match.end()` is the position *after* the punctuation
        end_pos = match.end()
        spans.append((last_start, end_pos))
        last_start = end_pos
    
    # If there's leftover text after the last match, include it
    if last_start < len(text):
        spans.append((last_start, len(text)))
    
    return spans


def split_cot_into_sentences(cot_text):
    """
    Return a list of substrings (each with exact raw text) representing "sentences".
    We do *not* alter or trim anything; each piece is directly from the original.
    """
    spans = find_sentence_spans(cot_text)
    # Extract the exact substrings
    sentences = [cot_text[start:end] for (start, end) in spans]
    return sentences


##################################
# 2. Fuzzy matching helpers
##################################

def fuzzy_match_sentence(target, sentence_list, cutoff=0.6):
    """
    Return the best match in `sentence_list` for the `target` string,
    or None if no match meets `cutoff`.
    """
    matches = difflib.get_close_matches(target, sentence_list, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def fuzzy_match_index(target, sentence_list, cutoff=0.6):
    """
    Return the index of the best match in `sentence_list` for the `target`,
    or None if no match meets `cutoff`.
    """
    match = fuzzy_match_sentence(target, sentence_list, cutoff=cutoff)
    if match is None:
        return None
    return sentence_list.index(match)


##################################
# 3. Annotating with <B> <EoB>
##################################

def annotate_hard_sentences(original_substrings, hard_indices_set):
    """
    Insert <B> and <EoB> around each substring (sentence) whose
    index is in `hard_indices_set`. We do NOT alter any other chars.
    
    Return a *single* string that is the concatenation of the original
    substrings with <B>/<EoB> inserted. 
    """
    annotated = []
    for i, chunk in enumerate(original_substrings):
        if i in hard_indices_set:
            annotated.append(f"<B>{chunk}<EoB>")
        else:
            annotated.append(chunk)
    # Simply join with no extra separator -> preserves all internal text as-is
    return "".join(annotated)


##################################
# 4. Call DeepSeek R1
##################################

import re

def call_deepseek_r1_for_difficult_sentences(cot_text, top_k, client, current_model):
    """
    Example using the DeepSeek R1 model via an OpenAI-compatible endpoint, 
    but using custom "tagged" delimiters for each sentence.

    - We ask the model to produce N 'hard' sentences, each wrapped as:

        -----SENTENCE START-----
        <verbatim content>
        -----SENTENCE END-----

    - Then we parse them using a regex.
    - If the model doesn't comply perfectly, we have a fallback to line-based splitting.
    
    Required external variables:
      client: an OpenAI-compatible client instance (e.g. `import openai` ...).
      current_model: the string name of the model (e.g. "deepseek-reasoner").

    Return:
      A list of top_k strings (each is a "hard" sentence).
    """
    # 1. Craft the system + user prompt
    system_message = f"""
You are DeepSeek-R1, a large reasoning model.
We will provide you with a chain-of-thought (CoT) text. 
We want the top {top_k} sentences that represent the most crucial or complex reasoning steps.

Important instructions:
1) Return EXACT text snippets verbatim from the CoT, including any newlines/tabs/punctuation.
2) For each sentence you return, enclose it in this format exactly:

-----SENTENCE START-----
<verbatim snippet>
-----SENTENCE END-----

3) Do not add commentary or extra formatting outside those tags.
4) If there are fewer than {top_k} total sentences, return them all.
5) Do not paraphrase or alter text in any way—verbatim means preserve whitespace and punctuation.
"""

    user_message = f"""\
Chain-of-thought (CoT): \n
{cot_text}
\n

Please list up to {top_k} of the hardest sentences. 
Use the exact tag format for each one:

-----SENTENCE START-----
<verbatim snippet>
-----SENTENCE END-----
"""

    # 2. Call DeepSeek R1 (OpenAI-compatible).
    response = client.chat.completions.create(
        model=current_model,
        messages=[
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": user_message.strip()}
        ],
        temperature=0.0,
    )

    raw_output = response.choices[0].message.content
    # import pdb; pdb.set_trace()

    # 3) Regex to capture variable dashes + "SENTENCE START/END"
    #    Explanation:
    #    -  -{3,} means "3 or more dashes"
    #    -  \s* means optional whitespace
    #    -  We allow multi-line inside (.*?) with DOTALL
    #    -  We'll remove leading/trailing whitespace around the snippet if desired
    #       or keep it exactly as is.
    pattern = (
        r"-{3,}\s*SENTENCE\s+START\s*-{3,}\s*"  # Opening tag with >=3 dashes
        r"(.*?)"                                # The captured snippet (non-greedy)
        r"\s*-{3,}\s*SENTENCE\s+END\s*-{3,}"    # Closing tag
    )

    matches = re.findall(pattern, raw_output, flags=re.DOTALL)

    # 4) Fallback if no matches
    if not matches:
        # Possibly the model didn't follow the tag format 
        # We'll do a naive line-based approach:
        lines = [ln.strip('\r') for ln in raw_output.split('\n') if ln.strip()]
        return lines[:top_k]
    
    # 5) If you want each snippet EXACTLY as captured, you can just return them. 
    #    If you prefer to remove leading/trailing newlines, do .strip() individually.
    snippets = [m for m in matches]

    # 6) Return up to top_k
    return snippets[:top_k]

##################################
# 5. Map function over dataset
##################################
def find_best_substring_match(snippet, text, min_ratio=0.5, chunk_size=100, overlap=50):
    """
    1) Try exact .find() to see if snippet is present verbatim in text.
       - If found, return that range immediately.
    2) If not found, do partial fuzzy matching but only within
       chunked windows of `chunk_size` length (with `overlap` offset).
       - This is faster than scanning the entire text with every possible start/end.
    3) Returns (best_start, best_end, best_ratio) or None if ratio < min_ratio.

    Args:
      snippet (str): The snippet we want to match in `text`.
      text (str): The large text where we look for the snippet.
      min_ratio (float): Minimum similarity ratio to accept a match.
      chunk_size (int): The size of each window in which we do naive scanning.
      overlap (int): Step size between successive windows 
                     (overlap = chunk_size // 2 is typical).

    Complexity:
      - If text is length N, we have about (N / (chunk_size - overlap)) windows.
      - Within each window of size ~chunk_size, we do a “sliding window” approach 
        around snippet_len, which may be up to ~20% on each side.  

    Tune chunk_size and overlap to get a good speed/accuracy tradeoff.
    """

    # 1) Fast exact check
    idx = text.find(snippet)
    if idx != -1:
        return (idx, idx + len(snippet), 1.0)  # perfect match

    # 2) Fuzzy search in chunked windows
    snippet_len = len(snippet)
    text_len = len(text)

    best_ratio = 0.0
    best_start = None
    best_end = None

    # We allow snippet length to vary from 80% to 120% of original
    lower_len = max(1, int(snippet_len * 0.8))
    upper_len = int(snippet_len * 1.2)

    # We'll iterate over chunk start indices with some overlap
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be larger than overlap")

    window_starts = range(0, text_len, step)

    for window_start in window_starts:
        window_end = min(window_start + chunk_size, text_len)
        window_text = text[window_start:window_end]
        window_text_len = len(window_text)

        # now do the naive approach in this chunk only
        for local_start in range(window_text_len):
            for size in range(lower_len, upper_len + 1):
                local_end = local_start + size
                if local_end > window_text_len:
                    break
                candidate = window_text[local_start:local_end]

                ratio = difflib.SequenceMatcher(None, snippet, candidate).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = window_start + local_start
                    best_end = window_start + local_end

    if best_ratio < min_ratio:
        return None
    return (best_start, best_end, best_ratio)


def insert_annotations(gen_text, matches):
    """
    Insert <B> and <EoB> tags into `gen_text` at the given 
    list of (start, end) offsets. 
    `matches` should be sorted in ascending order of start.

    For example, if you have matches = [(5,10), (20,25)] 
    then we want:
       gen_text[:5] + <B> + gen_text[5:10] + <EoB> + gen_text[10:20] 
         + <B> + gen_text[20:25] + <EoB> + gen_text[25:]
    """
    # We'll build the output in pieces
    annotated = []
    last_pos = 0

    for (start_idx, end_idx) in matches:
        # Add text before the match
        annotated.append(gen_text[last_pos:start_idx])
        # Add <B> + matched substring + <EoB>
        annotated.append(f"{START_BIGMODEL}{gen_text[start_idx:end_idx]}{END_BIGMODEL}")
        last_pos = end_idx

    # Add any leftover text after last match
    annotated.append(gen_text[last_pos:])
    return "".join(annotated)

def process_example(example, top_percent=0.4, llm_top_k=None):
    """
    For each generation in example["generations"]:
      1) Split into raw sentences (exact substrings).
      2) Decide how many we want the model to pick => `k`.
      3) Call the LLM (DeepSeek R1) to get the "hard" sentences (verbatim).
      4) Fuzzy-match them back to get their indices in the original list.
      5) Insert <B>, <EoB>.
    """
    new_annotated = []
    
    for gen_text in example["generations"]:
        # 1. Split into *exact* substrings
        sentences = split_cot_into_sentences(gen_text)
        num_sents = len(sentences)
        if num_sents == 0:
            # If for some reason there's no text, skip
            new_annotated.append(gen_text)
            continue
        
        # 2. Decide how many to request from the LLM
        if llm_top_k is not None:
            k = llm_top_k
        else:
            k = max(1, int(top_percent * num_sents))
        
        # 3. Query DeepSeek R1
        hard_candidates = call_deepseek_r1_for_difficult_sentences(gen_text, k, client, current_model)
        
        # 3. For each snippet, find the best fuzzy match in the entire gen_text
        match_offsets = []
        for hc in hard_candidates:
            best = find_best_substring_match(hc, gen_text, min_ratio=0.5)
            if best is not None:
                (start_idx, end_idx, ratio) = best
                match_offsets.append((start_idx, end_idx))

        # 4. Sort by start index, handle overlaps if needed
        match_offsets.sort(key=lambda x: x[0])

        # (Optional) merge or skip overlapping / almost contiguous matches:
        merged = []
        for (s, e) in match_offsets:
            if not merged:
                merged.append((s, e))
            else:
                old_s, old_e = merged[-1]
                # If the new interval starts after old_e + 5, it's sufficiently separate:
                if s > old_e + 10:
                    merged.append((s, e))
                else:
                    # Overlapping or contiguous with old_e
                    merged[-1] = (old_s, max(old_e, e))

        offload_chars = sum([x[1] - x[0] for x in merged])
        gen_text_tokens = len(tokenizer.tokenize(gen_text))
        offload_tokenset = [len(tokenizer.tokenize(gen_text[x[0]:x[1]])) for x in merged]
        offload_tokens = sum(offload_tokenset)

        total_chars = len(gen_text)
        print(f"Offloaded {offload_chars} / {total_chars} total chars\t Tokens: {offload_tokens} / {gen_text_tokens} \t\t Per-offload #Tokens: {offload_tokens / len(merged)}, \t\t Percentage Offload: {offload_chars / total_chars:.2%}")
        
        # 5. Insert <B>/<EoB>
        annotated_text = insert_annotations(gen_text, merged)
        new_annotated.append(annotated_text)

    example["annotated_generations"] = new_annotated
    return example
    #     # 4. Fuzzy match them to find indices
    #     hard_indices = set()
    #     for hc in hard_candidates:
    #         idx = fuzzy_match_index(hc, sentences, cutoff=0.5)
    #         if idx is not None:
    #             hard_indices.add(idx)
    #     import pdb; pdb.set_trace()
    #     # 5. Annotate
    #     annotated_text = annotate_hard_sentences(sentences, hard_indices)
    #     new_annotated.append(annotated_text)
    
    # example["annotated_generations"] = new_annotated
    # return example


##################################
# 6. Running it on the dataset
##################################

# def main():
# 1) Load dataset
ds = load_dataset("open-r1/OpenR1-Math-220k", "default")
print(ds)

# Just take 100 examples from the train split
ds_small = ds["train"].select(range(10))

# Now ds_small is a Dataset object containing only 100 rows
print(ds_small)

# 2) Configure your DeepSeek R1 API details (replace with real key/URL)
  # or set in .env file
# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_base="https://api.deepseek.com")'
# openai.api_base = "https://api.deepseek.com"   # or "https://api.deepseek.com/v1"

# 3) Process & annotate
# annotated_ds = ds.map(process_example, batched=False)
annotated_ds = ds_small.map(process_example, batched=False)

# 4) Now annotated_ds has an 'annotated_generations' column
# e.g., annotated_ds["train"][0]["annotated_generations"]
# will have the CoT text with <B>...<EoB> around the chosen "hard" sentences.

# 5) If you wish, save to disk
annotated_ds.save_to_disk("OpenR1_Math_Annotated")


# if __name__ == "__main__":
#     main()
