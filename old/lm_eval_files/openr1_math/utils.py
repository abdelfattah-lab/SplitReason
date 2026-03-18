"""
Utils for open-r1/OpenR1-Math-220k evaluation.
Uses math_verify for symbolic LaTeX verification (same as GRPO rewards).
Filters by correctness_llama (matching GRPO training), then subsamples 1000 examples.
"""
import importlib
import os
import re
from typing import Dict, List, Optional

import datasets
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

# Import doc_to_text and boxed helpers from sibling aime task directory
_aime_utils_path = os.path.join(os.path.dirname(__file__), '..', 'aime', 'utils.py')
_spec = importlib.util.spec_from_file_location("aime_utils", _aime_utils_path)
_aime_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aime_utils)

doc_to_text = _aime_utils.doc_to_text
last_boxed_only_string = _aime_utils.last_boxed_only_string
remove_boxed = _aime_utils.remove_boxed

SUBSAMPLE_N = 1000
SUBSAMPLE_SEED = 42


def _extract_boxed_answer(solution: str) -> Optional[str]:
    """Extract the content of the last \\boxed{} from a solution string."""
    boxed = last_boxed_only_string(solution)
    if boxed is not None:
        return remove_boxed(boxed)
    return None


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process OpenR1-Math docs: filter by correctness_llama, extract answer, then subsample."""
    # Filter to match GRPO training set
    original_len = len(dataset)
    dataset = dataset.filter(
        lambda ex: isinstance(ex.get("correctness_llama"), list)
        and len(ex["correctness_llama"]) > 0
        and all(ex["correctness_llama"]),
        num_proc=os.cpu_count(),
    )
    print(f"[openr1_math] Filtered by correctness_llama: {original_len} -> {len(dataset)}")

    def _process_doc(doc: dict) -> dict:
        problem = doc.get("problem", "")
        solution = doc.get("solution", "")
        # Prefer the existing 'answer' column; fall back to extracting from solution
        answer = doc.get("answer", None)
        if answer is None or answer == "":
            answer = _extract_boxed_answer(solution)
        if answer is None:
            print(f"Warning: Could not extract answer from doc: {problem[:200]}...")
            answer = ""
        return {
            "problem": problem,
            "solution": solution,
            "answer": answer,
        }

    dataset = dataset.map(_process_doc)

    # Random subsample
    if len(dataset) > SUBSAMPLE_N:
        dataset = dataset.shuffle(seed=SUBSAMPLE_SEED).select(range(SUBSAMPLE_N))
        print(f"[openr1_math] Subsampled {SUBSAMPLE_N} examples (seed={SUBSAMPLE_SEED})")

    return dataset


def process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Evaluate using math_verify (symbolic LaTeX verification)."""
    metrics = {"exact_match": None, "extracted_answers": []}

    gold = doc.get("answer", "")
    # Wrap gold in \boxed{} so math_verify can parse it
    gold_parsed = parse(
        f"\\boxed{{{gold}}}",
        extraction_mode="first_match",
    )

    if isinstance(results[0], list):
        results = results[0]

    for i, raw_pred in enumerate(results, 1):
        # Parse model answer using math_verify
        answer_parsed = parse(
            raw_pred,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )

        correct = 0.0
        if len(gold_parsed) and len(answer_parsed):
            try:
                correct = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                correct = 0.0

        metrics["extracted_answers"].append(str(answer_parsed))
        if i == 1:
            metrics["exact_match"] = correct

    return metrics
