# Speculative Reasoning

Put your HF_TOKEN and OPENAI_API_KEY in a .env file as:
```
OPENAI_API_KEY=XXXX
HF_TOKEN=XXXX
```

# Files

`speculative_reasoner.py` has flexible templates -- deprecate in favor of service.

`spec_service.py` is the service that can be run independently, or with lm-eval for evaluation of different methods.

# Installation instructions
`requirements.txt` will be added soon, currently should be pretty self explanatory.

We have to modify lm-evaluation-harness to (1) route requests to service (2) add tasks.

`git clone git@github.com:EleutherAI/lm-evaluation-harness.git`
`cd lm-evaluation-harness`
`git checkout 4cec66e4e468d15789473d6d63c3a61a751fa524`
`cd ./../`
`cp vllm_speculative.py lm-evaluation-harness/lm_eval/models/`
`cp vllm_speculative_init.py lm-evaluation-harness/lm_eval/models/__init__.py`
`cp tasks_init.py lm-evaluation-harness/lm_eval/tasks/__init__.py`
`cp -r aime lm-evaluation-harness/lm_eval/tasks/`
`cp -r openai_math lm-evaluation-harness/lm_eval/tasks/`
`cp -r openai lm-evaluation-harness/lm_eval/tasks/gpqa`
`cd lm-evaluation-harness`
`python -m pip install -e .[math,vllm]`
`cd ./../`

Credit to [s1](https://github.com/simplescaling/s1/tree/main) for lm-evaluation-harness modifications.

# Evaluate

`bash eval_script.sh`