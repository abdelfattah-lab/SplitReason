#!/usr/bin/env python3
"""DAPO entry point for datasets without correctness_llama column (e.g. competition_math).

Monkey-patches load_dataset in the dapo module so the correctness_llama
filter is silently skipped when the column doesn't exist.  Everything
else (rewards, trainer, logging) is identical to dapo.py.
"""

import open_r1.dapo as _dapo

# --- monkey-patch load_dataset in dapo's namespace ---
_orig_load = _dapo.load_dataset


def _load_skip_correctness_filter(*args, **kwargs):
    ds = _orig_load(*args, **kwargs)
    _orig_filter = ds.filter

    def _safe_filter(fn, **fkw):
        first_split = next(iter(ds.values()))
        if "correctness_llama" not in first_split.column_names:
            return ds
        return _orig_filter(fn, **fkw)

    ds.filter = _safe_filter
    return ds


_dapo.load_dataset = _load_skip_correctness_filter

# --- re-use the original main + CLI parsing ---
from open_r1.dapo import main  # noqa: E402
from open_r1.configs import DAPOConfig, GRPOScriptArguments  # noqa: E402
from trl import ModelConfig, TrlParser  # noqa: E402

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, DAPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
