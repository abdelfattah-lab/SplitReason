# SplitReason: Learning To Offload Reasoning

## Note: We are in the process of refactoring our code to change 'Speculative' to 'Split' to better align with the proposed methodology. The code still works as expected, and both Speculative / Split - named models are on huggingface.

![basic-image-describing-one-possible-reasoning-composition](./figs/accuracy_to_latency_teaser_main.png)
<!-- ![alt text](image.png) -->
This repository is in active development! Our primary focus is on Speculative Reasoning (our paper pdf is in this repository).

In this library, we aim to support several 'methods' of composing Large-Small reasoning language models to improve trade-off in reasoning quality - tokens generated.


For running evaluation, data-generation etc, put your HF_TOKEN and OPENAI_API_KEY in a .env file as:
```
OPENAI_API_KEY=XXXX
HF_TOKEN=XXXX
DEEPSEEK_API_KEY=XXXX
```

# Installation

`python -m pip install -r requirements.txt`

Additionally, please follow the setup commands below. We have to modify lm-evaluation-harness to:
- Route requests to our evaluation service (relies on vLLM).
- Add AIME24, MATH500 tasks for evaluation.

```
# Check-out appropriate LM-Evaluation-Harness commit ID.
git clone git@github.com:EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 4cec66e4e468d15789473d6d63c3a61a751fa524

cd ./../
# Add vllm_speculative and tasks.
cp lm_eval_files/vllm_speculative.py lm-evaluation-harness/lm_eval/models/
cp lm_eval_files/vllm_speculative_init.py lm-evaluation-harness/lm_eval/models/__init__.py
cp lm_eval_files/tasks_init.py lm-evaluation-harness/lm_eval/tasks/__init__.py
cp -r lm_eval_files/aime lm-evaluation-harness/lm_eval/tasks/
cp -r lm_eval_files/openai_math lm-evaluation-harness/lm_eval/tasks/
cp -r lm_eval_files/openai lm-evaluation-harness/lm_eval/tasks/gpqa

# Install package
cd lm-evaluation-harness
python -m pip install -e ".[math,vllm]"

cd ./../
```

Credit to [s1](https://github.com/simplescaling/s1/tree/main) for lm-evaluation-harness modifications.

# Training

We clone [open-r1](https://github.com/huggingface/open-r1) in `training/`. We also clone [trl](https://github.com/huggingface/trl/) as we conducted tests with / without adding 'special' tokens. 
For training, please follow the open-r1 installation instructions. 
**Environment for evaluation is different from training environment!**

`training/open-r1/` contains the base_training.sh script for SFT and GRPO. Please ensure you set all data and model file-paths correctly. Further, adjust your `recipes/accelerate_configs` based on number of available GPUs. 

As shown in `training/open-r1/base_training.sh` for GRPO, there are two scripts. These must be launched in parallel wiht the appropriate port configuration.

# Evaluate

Warning: there is a `fuser -k -9 /dev/nvidia*` in there, which will kill all your GPU jobs -- it is done to ensure everything is reset before loading new models.

`bash batch_eval_baselines.sh`


# Beyond Speculative Reasoning -- Adding new modes

This project started as an experimentation on how big-small models can work together to improve performance on reasoning tasks. We set up a simple evaluation framework that works relatively fast -- AIME24, MATH500 are good proxies for experimentation. We have reference baselines in `modes` such as `random_switch_flow.py`, which is the random-switching baseline from our paper. We also have `logprob_subselect.py`, which lets the small model do several parallel generations, then uses the big model for decoding a few tokens, and uses the decoded token log-probs to select promising decode-chains. While such experiments did not make it to the paper, it may be interesting for the open-source community to tinker with small-big language model compositions and strategies! To do so, please refer to the documentation below:

- Add args to spec_service.py and test_spec.py

Example (Random switching baseline):

```
    parser.add_argument("--random_switch", action="store_true")
    parser.add_argument("--switch_ratio", type=int, default=1, help="Switching ratio, always 1:{switch_ratio}")
    parser.add_argument("--switch_chunk", type=int, default=16)
```

- Parse args from data

```
    switch_ratio = data.get("switch_ratio", service_args.switch_ratio)
    switch_chunk = data.get("switch_chunk", service_args.switch_chunk)
```

- Add condition on speculative_reason function. Note that requests, batched_generate_text_vllmm batched_eval_logprob_vllm should be passed appropriately, as they are used. (Just pattern-match with `logprob_subselect_flow`)

```
    elif data.get("random_switch", False):
        final_reply, usage_data = run_random_switch_flow(
            # Mode related args here
                switch_ratio=switch_ratio,
                switch_chunk=switch_chunk,
            # Pretty much mandatory args below
                question=question,
                test_logging=test_logging,
                temperature=temperature,
                max_tokens=max_tokens,
                terminating_string=terminating_string,
                big_model_port=service_args.big_model_port,
                big_model=service_args.big_model,
                small_model_port=service_args.small_model_port,
                small_model=service_args.small_model,
                batched_generate_text_vllm=batched_generate_text_vllm,
                batched_eval_logprob_vllm=batched_eval_logprob_vllm,
                requests=requests
        )
```

- Add new args to `cmd = []` in test_spec.py as well as payload
```
            f"--switch_ratio={args.switch_ratio}",
            f"--switch_chunk={args.switch_chunk}",
```

```
    # Handle optional args as before
    if args.random_switch:
        cmd.append("--random_switch")
```

```
# In payload in main()
        "random_switch": args.random_switch,
        "switch_ratio": args.switch_ratio,
        "switch_chunk": args.switch_chunk,
```

- Add the args to `lm_eval_files/vllm_speculative.py` in `__init__`
```
        self.random_switch = self.service_params.get("random_switch", False)
        self.switch_ratio = self.service_params.get("switch_ratio", 1)
        self.switch_chunk = self.service_params.get("switch_chunk", 16)
```

- Use the above initialized values in `lm_eval_files/vllm_speculative.py` initializer in `_ensure_correct_service_is_running`
```

            f"--switch_ratio={self.switch_ratio}",
            f"--switch_chunk={self.switch_chunk}",
```

```
        if self.random_switch:
            cmd.append("--random_switch")
```

- Update lm-eval
```

cp lm_eval_files/vllm_speculative.py lm-evaluation-harness/lm_eval/models/

cd lm-evaluation-harness
python -m pip install -e .[math,vllm]
```

- Add assertion in test_spec.py
    - `if sum([args.placeholder_mode, args.spec_rewrite, args.logprob_subselect, args.big_model_only, args.small_model_only, args.random_switch]) != 1:`

- Create `random_switch_flow.py` in modes, code the `run_random_switch_flow` function, add import to `spec_service.py` 
    - `from modes.random_switch_flow import run_random_switch_flow`