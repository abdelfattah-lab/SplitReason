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

nlp = spacy.load("en_core_web_lg")  # or another language model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

load_dotenv("./../.env")
current_model = "deepseek-chat"
START_BIGMODEL = "<bigmodel>"
END_BIGMODEL = "<\\bigmodel>"

if current_model in ["deepseek-reasoner", "deepseek-chat"]:
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

##################################
# 1. Finding (start, end) spans
##################################

def find_sentence_spans_spacy(text):
    """
    Use spaCy to find sentence boundaries in `text`.
    Return a list of (start_index, end_index) for each sentence in the text.
    """
    doc = nlp(text)
    spans = []
    for sent in doc.sents:
        # sent.start_char and sent.end_char give the character offsets of the sentence
        spans.append((sent.start_char, sent.end_char))
    return spans


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
    # spans = find_sentence_spans_spacy(cot_text)
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


def call_deepseek_r1_for_difficult_snippets(cot_text, client, current_model, max_snippets=10):
    """
    Example using DeepSeek R1 model (OpenAI-compatible) to return 'difficult' 
    or 'hard' snippets from the chain-of-thought.
    
    - We ask the model to produce up to max_snippets "hardest" excerpts (short).
    - Each snippet is wrapped like this:

        -----SNIPPET START-----
        <verbatim content>
        -----SNIPPET END-----

    - Then we parse them via a regex.
    - If the model doesn't comply perfectly, we fallback to a naive line-based approach.

    Returns:
      A list of snippet strings, each containing the verbatim text from the CoT.
    """

    # 1) Craft the system + user prompt
    system_message = f"""
You are DeepSeek-R1, a large reasoning model.
We will provide you with a chain-of-thought (CoT) text.

We want you to identify and extract tricky, logically involved or complex portions.
Tricky parts can be 'difficult' reasoning sentences, equations, etc.
We want them as snippets (few lines, equations, etc.). 
Do NOT return large paragraphs. 
Focus on meaningful text that shows the hardest steps in the reasoning.

Important instructions:
1) Return up to {max_snippets} such 'difficult' snippets from the CoT.
2) Return each snippet verbatim (including newlines/punctuation).
3) For each snippet, enclose it **exactly** in:

-----SNIPPET START-----
<verbatim text>
-----SNIPPET END-----

4) No extra commentary or text before/between/after these tags. 
5) If there are fewer than {max_snippets} complex parts, return fewer.
6) Do not paraphrase or summarize: preserve exact whitespace and punctuation.
7) Return EXACT text snippets verbatim from the CoT, including any newlines/tabs/punctuation.
    """

    user_message = f"""\
Chain-of-thought (CoT): \n
{cot_text}
\n

Please produce up to {max_snippets} distinct short snippets for the most difficult or complex steps, 
verbatim, each wrapped in the specified tags.
Use the exact tag format for each one:
-----SNIPPET START-----
<verbatim text>
-----SNIPPET END-----
"""

    # 2) Make the completion request
    response = client.chat.completions.create(
        model=current_model,
        messages=[
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": user_message.strip()},
        ],
        temperature=0.0,
    )

    raw_output = response.choices[0].message.content

    # 3) Regex to capture the snippet blocks
    pattern = (
        r"-{3,}\s*SNIPPET\s+START\s*-{3,}\s*"
        r"(.*?)"
        r"\s*-{3,}\s*SNIPPET\s+END\s*-{3,}"
    )
    matches = re.findall(pattern, raw_output, flags=re.DOTALL)

    # 4) Fallback if the model did not use the tags
    if not matches:
        # fallback to naive line-based approach (or just return the entire output)
        lines = [ln.strip('\r') for ln in raw_output.split('\n') if ln.strip()]
        return lines[:max_snippets]

    # 5) Return each snippet exactly as captured.
    #    Optionally .strip() to remove leading/trailing newlines
    snippets = [m for m in matches]

    # 6) Return up to max_snippets
    return snippets[:max_snippets]


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
    # 1) Compute approximate fraction in terms of length
    #    This is just for reference in the prompt. 
    #    The model is responsible for matching it (or coming close).
    total_chars = len(cot_text)
    target_chars = int(total_chars * desired_fraction)
    # You might pass this info to the LLM to help it gauge how much text is ~20%.

    # 2) Craft the system prompt
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

    # max_attempts = 3
    # raw_output = None  # if it stays None, we never succeeded

    # # We'll construct the URL & headers for DeepSeek's "OpenAI-compatible" endpoint.
    # base_url = "https://api.deepseek.com/v1"  # recommended by DeepSeek docs
    # endpoint = "/chat/completions"
    # url = f"{base_url}{endpoint}"

    # headers = {
    #     "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",  # or whichever var
    #     "Content-Type": "application/json",
    # }

    # payload = {
    #     "model": current_model,
    #     "messages": [
    #         {"role": "system", "content": system_message},
    #         {"role": "user", "content": user_message},
    #     ],
    #     "temperature": 0.1,
    # }

    # for attempt in range(max_attempts):
    #     r = None
    #     raw_body = ""  # we'll capture any raw text

    #     try:
    #         r = httpx.post(url, headers=headers, json=payload, timeout=120)
    #         raw_body = r.text  # even if invalid JSON, we can store it

    #         if r.status_code == 200:
    #             # Attempt to parse JSON & extract the completion
    #             try:
    #                 data = r.json()  # can raise JSONDecodeError if invalid
    #                 raw_output = data["choices"][0]["message"]["content"]
    #             except:
    #                 # Write raw body to file for debugging
    #                 uid = f"{bidx}_{attempt}"
    #                 if not os.path.exists("attempts"):
    #                     os.makedirs("attempts")
    #                 with open(f"attempts/a_response_body+{uid}.txt", "w") as f:
    #                     f.write(raw_body)
    #                     f.write("\nEnd Of Response")
                
    #             # If it all worked, break out of retry loop
    #             break
    #         else:
    #             # If status != 200, treat it as "error" but don't crash
    #             uid = f"{bidx}_{attempt}"
    #             if not os.path.exists("attempts"):
    #                 os.makedirs("attempts")
    #             with open(f"attempts/system_message+{uid}.txt", "w") as f:
    #                 f.write(f"System message: {system_message}\n")
    #             with open(f"attempts/user_message+{uid}.txt", "w") as f:
    #                 f.write(f"User message: {user_message}\n")
    #             with open(f"attempts/response_body+{uid}.txt", "w") as f:
    #                 f.write(f"Status code: {r.status_code}\n\n")
    #                 f.write(raw_body)
    #                 f.write("\nEnd Of Response")
    #             print(f"\n[Loc-Index: {bidx}] [Attempt {attempt+1}/{max_attempts}] "
    #                 f"Non-200 status code: {r.status_code}")
                
    #     except Exception as e:
    #         # This covers JSONDecodeError, HTTP errors, etc.
    #         uid = f"{bidx}_{attempt}"
    #         if not os.path.exists("attempts"):
    #             os.makedirs("attempts")
    #         with open(f"attempts/system_message+{uid}.txt", "w") as f:
    #             f.write(f"System message: {system_message}\n")
    #         with open(f"attempts/user_message+{uid}.txt", "w") as f:
    #             f.write(f"User message: {user_message}\n")
    #         with open(f"attempts/response_body+{uid}.txt", "w") as f:
    #             f.write(f"Exception: {repr(e)}\n\n")
    #             if r is not None:
    #                 f.write(f"Status code: {r.status_code}\n\n")
    #                 f.write(raw_body)
    #                 f.write("\nEnd Of Response")
    #         print(f"\n[Loc-Index: {bidx}] [Attempt {attempt+1}/{max_attempts}] "
    #             f"Exception: {e}")

    #     # If we haven’t broken out yet, it means we didn’t succeed.
    #     # Sleep if not on last attempt:
    #     if attempt < max_attempts - 1 and raw_output is None:
    #         time.sleep(60)

    # After the loop, raw_output is either the model's text (if we got a success)
    # or None (if all attempts failed).

    max_attempts = 1
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
            )
            # If we got here, call succeeded; break out of the retry loop
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
            # Print debugging info
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


# def find_best_substring_match_simple(snippet, text, min_ratio=0.5):
#     best_ratio = 0.0
#     best_start = None
#     best_end = None

#     snippet_len = len(snippet)
#     text_len = len(text)

#     # If the snippet is longer than the text, no point searching.
#     if snippet_len > text_len:
#         return None

#     for i in range(text_len - snippet_len + 1):
#         candidate = text[i : i + snippet_len]
#         ratio = difflib.SequenceMatcher(None, snippet, candidate).ratio()
#         if ratio > best_ratio:
#             best_ratio = ratio
#             best_start = i
#             best_end = i + snippet_len

#     if best_ratio >= min_ratio:
#         return (best_start, best_end, best_ratio)
#     else:
#         return None


def find_best_substring_match_simple(
    snippet, 
    text, 
    min_ratio=0.5, 
    refine_threshold=0.99, 
    boundary_search_size=50,
    boundary_sub_len=5
):
    """
    Find the fuzzy substring match of `snippet` in `text` by checking
    all possible positions. If the best ratio >= min_ratio, return
    (start, end, ratio). Otherwise return None.
    
    Then optionally refine boundaries if ratio < refine_threshold.
    
    Args:
        snippet (str): The snippet to match.
        text (str): The larger text in which we search.
        min_ratio (float): Minimum fuzzy match ratio to accept a match.
        refine_threshold (float): If best_ratio < this, we try to refine boundaries.
        boundary_search_size (int): The +/- range around best_start/best_end to search
                                    for an exact boundary match.
        boundary_sub_len (int): How many characters (front & back of snippet) to look for exactly.
    
    Returns:
        (best_start, best_end, best_ratio) or None
    """
    snippet_len = len(snippet)
    text_len = len(text)
    
    # If snippet longer than text, no point
    if snippet_len > text_len:
        return None
    
    # We'll do all fuzzy checking in lowercase
    snippet_lower = snippet.lower()
    text_lower = text.lower()
    
    best_ratio = 0.0
    best_start = None
    best_end = None
    
    # 1) Find the best fuzzy match in one pass
    for i in range(text_len - snippet_len + 1):
        candidate = text_lower[i : i + snippet_len]
        ratio = difflib.SequenceMatcher(None, snippet_lower, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
            best_end = i + snippet_len
    
    # If we never found anything above min_ratio, return None
    if best_ratio < min_ratio:
        return None
    
    # 2) If ratio is below some "refine" threshold, 
    #    try to find "cleaner" boundaries near best_start, best_end
    if best_ratio < refine_threshold:
        # We'll search for the first N=5 chars and last N=5 chars of snippet (by default)
        start_sub = snippet_lower[:boundary_sub_len]
        end_sub   = snippet_lower[-boundary_sub_len:]
        
        # --- Refine the start ---
        if start_sub:
            # Search region for the start substring
            left_bound = max(0, best_start - boundary_search_size)
            right_bound = min(text_len, best_start + boundary_search_size)
            
            # Attempt to find `start_sub` in text_lower[left_bound : right_bound]
            offset_region = text_lower[left_bound : right_bound]
            local_idx = offset_region.find(start_sub)
            
            if local_idx != -1:
                # new_start is offset from `left_bound`
                new_start = left_bound + local_idx
                # Only update if this new start is "close" to best_start
                if abs(new_start - best_start) <= boundary_search_size:
                    best_start = new_start
        
        # --- Refine the end ---
        if end_sub:
            # Search region for the end substring
            left_bound = max(0, best_end - boundary_search_size - boundary_sub_len)
            right_bound = min(text_len, best_end + boundary_search_size)
            
            offset_region = text_lower[left_bound : right_bound]
            local_idx = offset_region.find(end_sub)
            
            if local_idx != -1:
                new_end = left_bound + local_idx + boundary_sub_len
                if abs(new_end - best_end) <= boundary_search_size:
                    best_end = new_end
    
    # Return the final best match
    return (best_start, best_end, best_ratio)


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
        # print("--" * 20)
        # print(f"Appending \n {gen_text[start_idx:end_idx]} \n")
        # print("--" * 20)
        annotated.append(f"{START_BIGMODEL}{gen_text[start_idx:end_idx]}{END_BIGMODEL}")
        last_pos = end_idx

    # Add any leftover text after last match
    annotated.append(gen_text[last_pos:])
    return "".join(annotated)



def process_example(example, bidx=None, top_percent=0.4, llm_top_k=None):
    """
    For each generation in example["generations"]:
      1) Split into raw sentences (exact substrings).
      2) Decide how many we want the model to pick => `k`.
      3) Call the LLM (DeepSeek R1) to get the "hard" sentences (verbatim).
      4) Fuzzy-match them back to get their indices in the original list.
      5) Insert <B>, <EoB>.
    """
    new_annotated = []
    
    for genid, gen_text in enumerate(example["generations"]):
        # # 1. Split into *exact* substrings
        # sentences = split_cot_into_sentences(gen_text)
        # num_sents = len(sentences)
        # # pprint(sentences)
        # for i, sent in enumerate(sentences):
        #     print("--" * 20)
        #     print(sent)
        #     print("--" * 20)
        # print("\n\n", "--" * 20, "\n\n")
        # if num_sents == 0:
        #     # If for some reason there's no text, skip
        #     new_annotated.append(gen_text)
        #     continue
        # import pdb; pdb.set_trace()
        # # 2. Decide how many to request from the LLM
        # if llm_top_k is not None:
        #     k = llm_top_k
        # else:
        #     k = max(1, int(top_percent * num_sents))
        try:
            # 3. Query DeepSeek R1
            # hard_candidates = call_deepseek_r1_for_difficult_snippets(gen_text, client, current_model, max_snippets=20)
            hard_candidates = call_deepseek_r1_for_difficult_snippets_fraction(gen_text, bidx, client, current_model, desired_fraction=0.2)
            # call_deepseek_r1_for_difficult_snippets_fraction
            # 3. For each snippet, find the best fuzzy match in the entire gen_text
            match_offsets = []
            for hc in hard_candidates:
                # best = find_best_substring_match(hc, gen_text, min_ratio=0.5)
                best = find_best_substring_match_simple(hc, gen_text, min_ratio=0.5)
                if best is not None:
                    (start_idx, end_idx, ratio) = best
                    match_offsets.append((start_idx, end_idx))
            # 4. Sort by start index, handle overlaps if needed
            match_offsets.sort(key=lambda x: x[0])
            # if len(match_offsets) == 0:
            #     import pdb; pdb.set_trace()

            print("Match intervals:\t", match_offsets)
            # (Optional) merge or skip overlapping / almost contiguous matches:
            merged = []
            for (s, e) in match_offsets:
                if not merged:
                    merged.append((s, e))
                else:
                    old_s, old_e = merged[-1]
                    # If the new interval starts after old_e + 10, it's sufficiently separate:
                    if s > old_e + 10:
                        merged.append((s, e))
                    else:
                        # Overlapping or contiguous with old_e
                        merged[-1] = (old_s, max(old_e, e))
            print("Merged intervals:\t", merged)
            offload_chars = sum([x[1] - x[0] for x in merged])
            gen_text_tokens = len(tokenizer.tokenize(gen_text))
            offload_tokenset = [len(tokenizer.tokenize(gen_text[x[0]:x[1]])) for x in merged]
            offload_tokens = sum(offload_tokenset)

            total_chars = len(gen_text)
            offloading_characteristic = f"{bidx}:\tOffloaded {offload_chars} / {total_chars} total chars\t Tokens: {offload_tokens} / {gen_text_tokens} \t\t Per-offload #Tokens: {offload_tokens / len(merged)}, \t\t Percentage Offload: {offload_chars / total_chars:.2%}"
            print(offloading_characteristic)
            # append fields from offloading_characteristic to a file as a csv 
            ex_uuid = example["uuid"]
            if not os.path.exists("offloading_characteristic.csv"):
                with open("offloading_characteristic.csv", "w") as f:
                    f.write("uuid,genid,offload_chars,total_chars,offload_tokens,gen_text_tokens,offload_tokens_per_sentence,percentage_offload\n")
            with open("offloading_characteristic.csv", "a") as f:
                f.write(f"{ex_uuid},{genid},{offload_chars},{total_chars},{offload_tokens},{gen_text_tokens},{offload_tokens / len(merged)},{offload_chars / total_chars:.2%}\n")
            # write merged as a string to csv with uuid
            with open("merged_intervals.csv", "a") as f:
                f.write(f"{ex_uuid},{merged}\n")
            # 5. Insert <B>/<EoB>
            annotated_text = insert_annotations(gen_text, merged)
            new_annotated.append(annotated_text)
        except Exception as e:
            # If the LLM fails, we can just skip this example
            import traceback
            traceback.print_exc()
            print(f"LocIDX {bidx} Failed: Error processing example\n{e}")
            new_annotated.append(f"{START_BIGMODEL}Error processing Due To {e} \n {END_BIGMODEL}")

    example["annotated_generations"] = new_annotated
    return example




def process_batch_in_parallel(batch, top_percent=0.4, llm_top_k=None):
    annotated_batch = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
        # schedule jobs
        futures = []
        for bidx, ex in enumerate(batch):
            futures.append(executor.submit(process_example, ex, bidx, top_percent, llm_top_k))
        
        # collect results as they finish
        for f in concurrent.futures.as_completed(futures):
            annotated_example = f.result()
            annotated_batch.append(annotated_example)
    # import pdb; pdb.set_trace()
    
    return annotated_batch

ds = load_dataset("open-r1/OpenR1-Math-220k", "default")

# ds_small = ds["train"].select(range(20))
# select index 400 to 420
# ds_small = ds["train"].select(range(400, 420))
ds_small = ds["train"].select(range(5500))

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
        if any(length < 200 for length in lens):
            print(f"Discarding example {idx} with lengths {lens}")
            processed_indices.discard(idx)
            to_remove.append(idx)
    keep_indices = [i for i in range(len(partial_dataset)) if i not in to_remove]
    partial_dataset = partial_dataset.select(keep_indices)
    all_results = partial_dataset.to_list()
else:
    partial_dataset = None
    all_results = []

num_parallel = 200
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

##################################
# 6. Running it on the dataset
##################################

    # # def main():
    # # 1) Load dataset
    # ds = load_dataset("open-r1/OpenR1-Math-220k", "default")

    # # Just take 100 examples from the train split
    # ds_small = ds["train"].select(range(10))

    # # Now ds_small is a Dataset object containing only 100 rows
    # print(ds_small)

    # # 3) Process & annotate
    # annotated_ds = ds_small.map(process_example, batched=True)

    # # 5) If you wish, save to disk
    # if current_model in ["deepseek-reasoner", "deepseek-chat"]:
    #     annotated_ds.save_to_disk("OpenR1_Math_Annotated_DeepSeek")
    # else:
    #     annotated_ds.save_to_disk("OpenR1_Math_Annotated_GPT")

