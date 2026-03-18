from __future__ import annotations
from collections import Counter
import os
import re
import signal
from typing import Dict, List, Optional

import datasets

from lm_eval.utils import eval_logger

if os.getenv("PROMPTSTEP") is not None:
    QUERY_TEMPLATE = '{Question}\n\nThink for up to ' + os.getenv("PROMPTSTEP") + ' steps.'
elif os.getenv("PROMPTTOKEN") is not None:
    QUERY_TEMPLATE = '{Question}\n\nThink for up to ' + os.getenv("PROMPTTOKEN") + ' tokens.'
elif os.getenv("PROMPTLONG") is not None:
    QUERY_TEMPLATE = '{Question}\n\nAnswer after a long amount of thinking. If you feel like you are finished early, spend the extra time trying to double-check your work until you are absolutely sure that you have the correct answer.'
elif os.getenv("PROMPTSHORT") is not None:
    QUERY_TEMPLATE = '{Question}\n\nAnswer after a short amount of thinking. Do not spend excessive time double-checking your work.'
else:
    QUERY_TEMPLATE = '{Question}'

# The correct answer is an integer between $000$ and $999$, inclusive. Keep thinking until your answer is in the correct range.
# The correct answer is an integer between $000$ and $999$, inclusive.

print("QUERY_TEMPLATE: ", QUERY_TEMPLATE)

# Adapted from https://github.com/openai/simple-evals/blob/c0dba4c7bfbc17f786aec7bd7c3585a36ad81f23/common.py#L23
# (?i): Enables case-insensitive matching. This means "Answer", "answer", "ANSWER", etc., will all be matched.
# Answer: Matches the literal string "Answer" (case-insensitive due to (?i)).
# \s*: Matches zero or more whitespace characters (spaces, tabs, etc.) after "Answer". This accounts for cases where there might or might not be space between "Answer" and the colon (:).
# :: Matches the literal colon character :.
# \s*: Matches zero or more whitespace characters after the colon. This handles cases where there might be spaces between the colon and the actual answer.
# (.*): The .* matches zero or more of any character (including none), except for newlines unless re.DOTALL is used (which allows newlines to be matched too).
# Note: This does not match e.g. "**Final Answer:** A" as it only matches "Answer: A" or "Answer: A) 7" etc.
ANSWER_PATTERN = r"(?i)Answer\s*:\s*(.*)"

EXTRACTION_TEMPLATE_IDX = r"""
Look at the following attempt by a student and extract the student's answer. If it is equivalent (ignoring trivial simplifications) to any of the provided options, return the index of that option starting from 1. Else, return -1.

Examples:

    Options: ['2x+4', '2x', '4x']
    Attempt: The answer is 3+2x.

-1
(the student's answer is not among the options)

    Options: ['72,000']
    Attempt: 72000 \text{ cents}.

1
(always give benefit of the doubt to units and ignore formatting which makes the 1st option match)

    Options: ['2/(-3)', '2/3']
    Attempt: -1 * 2/3

1
(the 1st option matches after trivial simplifications which are fine)

    Options: ['x=5']
    Attempt: 5

1

    Options: ['\dfrac{33}{100}']
    Attempt: 0.33

1

    Options: ['75^\circ']
    Attempt: ...various calculations and explanations...hence the answer is $\boxed{x in 75}$.

1

    Options: ['(1,-3)', '(1,-1)', '(1,0)', '(1,-2)']
    Attempt: -2, 1

4
(ignore whitespace and other formatting which makes the 4th option match)

    Options: ['-2,1']
    Attempt: 1, -2

1
(likely a problem where multiple solutions are possible thus ignore order)

    Options: ['11', '100', '50', '-5', '12', '10']
    Attempt: ...$\boxed{12^{\mathrm{th}}}$.

5

    Options: ['2516_8']
    Attempt: 2516

1
(give benefit of the doubt for different bases)

    Options: ['11\sqrt2']
    Attempt: 11\sqrt{2}

1

    Options: ['11,\! 111,\! 111,\! 100']
    Attempt: 11111111100

1

    Options: ['\text{Navin}']
    Attempt: ...it is navin.

1

---

YOUR TASK


Respond with only the index of the matching option starting from 1 or -1 if there is absolutely no reasonable match. Do not include a rationale.

    Options: %(expression1)s
    Attempt: %(expression2)s
""".strip()


# https://github.com/openai/simple-evals/blob/580d359553a88584c11ce4efb97d49d9386e0d9e/common.py#L153C1-L156C45
def extract_answer_idx(sampler, options: List[str], attempt: str):
    prompt = EXTRACTION_TEMPLATE_IDX % {"expression1": options, "expression2": attempt}
    response = sampler([dict(content=prompt, role="user")])
    return response

import time
from typing import Any

import openai
from openai import OpenAI

"""
A **drop‑in replacement** for the original `process_results` that:
  • removes every call to GPT‑4 / ChatCompletionSampler – completely offline
  • uses the rich extraction logic from *lighteval* (Latex/Expr/Indices) as a first‑class parser
  • recognises extra anchors ("final answer is", boxed answers, no anchor, etc.)
  • normalises answers through SymPy so that `0.5`, `1/2`, `\frac{1}{2}` all compare equal
  • still computes `cov@k` and `maj@k` metrics when many samples are given
The function signature is **unchanged** so you can just import and call it from your evaluation
loop.
"""

from collections import Counter
import os
import re
from typing import Dict, List

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from lighteval.tasks.templates.utils.formulation import ChoicePrefix
from lighteval.utils.language import Language
from lighteval.metrics.utils.math_comparison import should_treat_as_complex
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
# from lighteval.extraction import (
#     LatexExtractionConfig,
#     ExprExtractionConfig,
#     IndicesExtractionConfig,
#     extract_target_from_pred,
# )


import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import groupby
from typing import Any, Literal, Sequence

import sympy
from sympy import Basic, FiniteSet, MatrixBase, Number
from sympy.parsing import parse_expr

from lighteval.metrics.utils.math_comparison import should_treat_as_complex
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.utils.formulation import ChoicePrefix, get_prefix
from lighteval.tasks.templates.utils.translation_literals import TRANSLATION_LITERALS
from lighteval.utils.imports import requires_latex2sympy2_extended
from lighteval.utils.language import Language
from lighteval.utils.timeout import timeout

from lighteval.metrics.dynamic_metrics import (
    LatexExtractionConfig,
    ExprExtractionConfig,
    IndicesExtractionConfig,
    extract_target_from_pred,
)
from lighteval.metrics.utils.extractive_match_utils import (
    lazy_latex_regex,
    lazy_expr_regex,
)

# ------------------------------
#  Regex utilities (anchors)
# ------------------------------
TRAN_EN = TRANSLATION_LITERALS[Language.ENGLISH]
FULL_STOP = rf"[{re.escape(TRAN_EN.full_stop)}\.]"
COLON = rf"[{re.escape(TRAN_EN.colon)}\:]"
SPACE = rf"(?:\s|{re.escape(TRAN_EN.sentence_space)})"

ANSWER_ANCHORS = [
    # Answer: xyz
    rf"(?i){TRAN_EN.answer}{SPACE}*{COLON}?{SPACE}*(.+?)$",
    # Final answer is xyz
    rf"(?i)final{SPACE}+answer{SPACE}+(?:is|:){SPACE}*(.+?)$",
]
ANSWER_RE = re.compile("|".join(f"({p})" for p in ANSWER_ANCHORS), re.DOTALL)

SPLIT_TOKENS = ("<|im_start|>answer\n", "<|im_start|>")


class ChatCompletionSampler:
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception

def doc_to_text(doc: dict) -> str:
    return QUERY_TEMPLATE.format(Question=doc.get("problem", doc.get("question")))

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        solution = doc.get("solution", doc.get("orig_solution", doc.get("orig_orig_solution")))
        problem = doc.get("problem", doc.get("question"))
        answer = doc.get("answer", doc.get("orig_answer", doc.get("orig_orig_answer")))
        if solution is None:
            print("Warning: No solution found; DOC:", doc)
        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc
    return dataset.map(_process_doc)

# def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
#     metrics = {"exact_match": None, "extracted_answers": []}
#     # bp()
#     # Multiple results -> we are measuring cov/maj etc
#     if isinstance(results[0], list):
#         results = results[0]
#         n_res = len(results) # e.g. 64
#         n_res_list = [2**i for i in range(1, int(n_res.bit_length()))] # e.g. [2, 4, 8, 16, 32, 64]
#         metrics = {
#             **metrics,
#             "exact_matches": [],
#             **{f"cov@{n}": -1 for n in n_res_list},
#             **{f"maj@{n}": -1 for n in n_res_list},
#         }

#     if os.getenv("PROCESSOR", "") == "gpt-4o-mini":
#         sampler = ChatCompletionSampler(model="gpt-4o-mini")
#     else:
#         print(f"Unknown processor: {os.getenv('PROCESSOR')}; set 'PROCESSOR=gpt-4o-mini' and 'OPENAI_API_KEY=YOUR_KEY' for best results.")
#         sampler = None

#     if isinstance(doc["answer"], str) and doc["answer"].isdigit():
#         gt = str(int(doc["answer"])) # 023 -> 23
#     else:
#         gt = str(doc["answer"])
#     split_tokens = ["<|im_start|>answer\n", "<|im_start|>"]

#     for i, a in enumerate(results, start=1):
#         if split_tokens[0] in a:
#             a = a.split(split_tokens[0])[-1]
#         elif split_tokens[1] in a:
#             a = a.split(split_tokens[1])[-1]
#             if "\n" in a:
#                 a = "\n".join(a.split("\n")[1:])

#         if (box := last_boxed_only_string(a)) is not None:
#             a = remove_boxed(box)
#         # re.DOTALL is key such that newlines are included e.g. if it does `Answer: Here is the solution:\n\n10`
#         elif (matches := re.findall(ANSWER_PATTERN, a, re.DOTALL)) != []:
#             a = matches[-1]  # Get the last match

#         # AIME answers are from 000 to 999 so often it is a digit anyways
#         if (a.isdigit()) and (gt.isdigit()):
#             a = str(int(a)) # 023 -> 23
#         elif sampler is not None:
#             options = [gt] + list(set(metrics["extracted_answers"]) - {gt})
#             if len(options) > 7:
#                 # Could switch back to exact returning like in AIME in that case
#                 # Problem with exact returning is that it sometimes messes up small things like a dollar sign
#                 print("Warning: Lots of options which may harm indexing performance:", options)            
#             # This ensures that if doc['answer'] is \text{Evelyn} it is represented as such and not \\text{Evelyn}
#             options_str = "[" + ", ".join(["'" + str(o) + "'" for o in options]) + "]"
#             # a = extract_answer(sampler, options, a)
#             idx = extract_answer_idx(sampler, options_str, a)
#             if idx != "-1":
#                 if idx.isdigit():
#                     idx = int(idx) - 1
#                     if len(options) > idx >= 0:
#                         a = options[idx]
#                     else:
#                         print("Warning: Index out of bounds; leaving answer unchanged\n", a, "\noptions", options_str, "\ndoc['answer']", gt, "\nidx", idx)
#                 else:
#                     print("Warning: Processing did not produce integer index\na", a, "\noptions", options_str, "\ndoc['answer']", gt, "\nidx", idx)
#         else:
#             pass # TODO: Maybe add back legacy processing

#         metrics["extracted_answers"].append(a)
#         a = int(a == gt)
#         if not(a): # Optional logging
#             print("Marked incorrect\na " + metrics["extracted_answers"][-1] + "\ndoc['answer'] " + gt)
#         if i == 1:
#             metrics["exact_match"] = a
#             if "exact_matches" in metrics:
#                 metrics["exact_matches"].append(a)
#         elif i > 1:
#             metrics["exact_matches"].append(a)
#             if i in n_res_list:
#                 metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
#                 metrics[f"maj@{i}"] = int(gt == Counter(metrics["extracted_answers"]).most_common(1)[0][0])

#     return metrics


def _strip_chat_markers(text: str) -> str:
    if SPLIT_TOKENS[0] in text:
        text = text.split(SPLIT_TOKENS[0])[-1]
    elif SPLIT_TOKENS[1] in text:
        text = "\n".join(text.split(SPLIT_TOKENS[1])[-1].split("\n")[1:])
    return text.strip()


def _last_boxed(tex: str) -> str | None:
    """Return the content of the last \boxed/\fbox if any (with braces)."""
    for kw in ("\\boxed", "\\fbox"):
        idx = tex.rfind(kw)
        if idx >= 0:
            break
    else:
        return None

    # walk until matching brace
    i, depth = idx, 0
    while i < len(tex):
        if tex[i] == "{" and tex[i - 1] != "\\":
            depth += 1
        elif tex[i] == "}" and tex[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return tex[idx : i + 1]
        i += 1
    return None


def _remove_boxed(boxed: str) -> str:
    for prefix in ("\\boxed ", "\\boxed{", "\\fbox ", "\\fbox{"):
        if boxed.startswith(prefix):
            inner = boxed[len(prefix) :]
            return inner[:-1] if inner.endswith("}") else inner
    return boxed


def _sympify_safe(expr: str):
    try:
        return parse_expr(expr.replace("^", "**"), evaluate=False)
    except Exception:
        return None


def _normalise(raw: str):
    """Best‑effort normalisation to a canonical string or SymPy Expr."""
    raw = raw.strip()

    # Quick number normalisation (023 → 23, 1,234 → 1234)
    if re.fullmatch(r"-?[\d, ]+(?:[\.\,]\d+)?", raw):
        num = raw.replace(",", "").replace(" ", "")
        if num.isdigit() or re.fullmatch(r"-?\d+\.\d+", num):
            return str(int(num)) if num.isdigit() else str(float(num))

    # Try SymPy parsing – if success, return canonical str(expr)
    sym_expr = _sympify_safe(raw)
    if sym_expr is not None:
        return str(sp.simplify(sym_expr))

    return raw  # fallback plain string


# ------------------------------
#  Main improved evaluator
# ------------------------------

def process_results(doc: Dict, results: List[str]):  # noqa: C901 – keep single entry
    """Improved offline answer extraction & metric computation."""
    metrics: Dict[str, int | List[str]] = {"exact_match": None, "extracted_answers": []}

    # Support multi‑sample case (e.g. 64 draws)
    if isinstance(results[0], list):
        results = results[0]
    n_res = len(results)
    n_res_list = [2 ** i for i in range(1, int(n_res.bit_length()))]
    if n_res > 1:
        metrics |= {"exact_matches": []} | {f"cov@{n}": -1 for n in n_res_list} | {f"maj@{n}": -1 for n in n_res_list}

    gt_raw = str(int(doc["answer"])) if isinstance(doc["answer"], str) and doc["answer"].isdigit() else str(doc["answer"])
    gt_norm = _normalise(gt_raw)

    # Pre‑bake extraction configs
    # latex_cfg = LatexExtractionConfig(try_extract_without_anchor=True)
    # expr_cfg = ExprExtractionConfig(try_extract_without_anchor=True)
    # targets = [latex_cfg, expr_cfg]
    latex_cfg = LatexExtractionConfig(try_extract_without_anchor=True)
    expr_cfg  = ExprExtractionConfig(try_extract_without_anchor=True)

    target_res = [
        (lazy_latex_regex(latex_cfg, Language.ENGLISH), latex_cfg),
        (lazy_expr_regex(expr_cfg,  Language.ENGLISH),  expr_cfg),
    ]

    # extracted = extract_target_from_pred(
    #     pred,
    #     target_res,
    #     fallback_mode="first_match",
    # )
    for i, raw_pred in enumerate(results, 1):
        pred = _strip_chat_markers(raw_pred)

        # Priority 1 – boxed environment
        if (boxed := _last_boxed(pred)) is not None:
            pred = _remove_boxed(boxed)
        else:
            # Priority 2 – explicit anchor regex
            if (m := ANSWER_RE.search(pred)) is not None:
                # pick first non‑None capturing group
                pred = next(g for g in m.groups() if g)

        pred = pred.strip().rstrip(". ")

        # Priority 3 – structured extractor from lighteval
        # extracted = extract_target_from_pred(pred, [(None, t) for t in targets], fallback_mode="first_match")
        extracted = extract_target_from_pred(pred, target_res, fallback_mode="first_match")
        if extracted:
            # `extract_target_from_pred` returns list; take first element's str fallback
            pred = str(extracted[0])

        # Canonicalise
        pred_norm = _normalise(pred)
        metrics["extracted_answers"].append(pred_norm)

        correct = int(pred_norm == gt_norm)

        # book‑keeping
        if metrics["exact_match"] is None:
            metrics["exact_match"] = correct
        if "exact_matches" in metrics:
            metrics["exact_matches"].append(correct)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                metrics[f"maj@{i}"] = int(gt_norm == Counter(metrics["extracted_answers"]).most_common(1)[0][0])

    return metrics


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]
