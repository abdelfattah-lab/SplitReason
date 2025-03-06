# Speculative Reasoning

![basic-image-describing-one-possible-reasoning-composition](./figs/image.png)
In this library, we aim to support several 'methods' of composing Large-Small reasoning language models to improve trade-off in reasoning quality - tokens generated.


For running evaluation, put your HF_TOKEN and OPENAI_API_KEY in a .env file as:
```
OPENAI_API_KEY=XXXX
HF_TOKEN=XXXX
```

# Installation

`python -m pip install -r requirements.txt`

Additionally, please follow the setup commands below. We have to modify lm-evaluation-harness to:
- Route requests to our evaluation service.
- Add AIME24 and MATH tasks for evaluation.

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
python -m pip install -e .[math,vllm]

cd ./../
```

Credit to [s1](https://github.com/simplescaling/s1/tree/main) for lm-evaluation-harness modifications.

# Evaluate

`bash eval_script.sh`

# Test with custom questions

As an example, this will sequentially scale once, re-draft the CoT before the sequential scaling, re-write the full CoT everytime. Also it will encourage the small model to propose potential reasoning errors for continuity. 

`python test_spec.py --test_logging --thinking_n_ignore 1 --drafting_n 1 --full_rewrite --draft_propose_ignore_str`