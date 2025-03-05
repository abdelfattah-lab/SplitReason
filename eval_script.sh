#!/usr/bin/env bash

export $(grep -v '^#' .env | xargs)

export VLLM_WORKER_MULTIPROC_METHOD='spawn'
export PROCESSOR=gpt-4o-mini

python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,
    pretrained=meta-llama/Llama-2-7b-chat-hf,
    big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,
    big_model_port=8000,
    big_model_gpus=0|1,
    small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,
    small_model_port=8001,
    small_model_gpus=2,
    thinking_n_ignore=1,
    drafting_n=0,
    full_rewrite=True,
    bloat_tokens=0,
    max_tokens=16384,
    terminate_on_exit=True" \
    --tasks aime24_nofigures \
    --batch_size auto --apply_chat_template \
    --output_path nottc --log_samples --gen_kwargs max_gen_toks=2048