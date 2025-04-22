#!/usr/bin/env bash

export $(grep -v '^#' .env | xargs)

export VLLM_WORKER_MULTIPROC_METHOD='spawn'
export PROCESSOR=gpt-4o-mini
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=logging_config.json


# Note, it is best to average / median results over several generations :) 
# Most pass@1 do 16 generations, but we rely on simply repeating the experiment 5 times due to compute constraints.


fuser -k -9 /dev/nvidia*
# Test to have 32B big model, 1.5B small model
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_32B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_32B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_32B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_32B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_32B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

fuser -k -9 /dev/nvidia*
     
# Test to have 14B big model, 1.5B small model
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_14B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_14B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_14B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_14B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_14B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

fuser -k -9 /dev/nvidia*     

# Test to have 14B big model, 1.5B small model
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 