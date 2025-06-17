#!/usr/bin/env bash

export $(grep -v '^#' .env | xargs)

export VLLM_WORKER_MULTIPROC_METHOD='spawn'
export PROCESSOR=gpt-4o-mini
# export VLLM_CONFIGURE_LOGGING=1
# export VLLM_LOGGING_CONFIG_PATH=logging_config.json

# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason_perf=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0,small_model_gpus=1,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#      --tasks aime24_nofigures  --limit 30 --batch_size auto --apply_chat_template  --output_path log_traces/DEBUG_10June --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>"  

# akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-E2EGRPO-OpenR1_Math_SpecR_GRPO_Mini-MiniSet


python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason_perf=True    ,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-E2EGRPO-OpenR1_Math_SpecR_GRPO_Mini-MiniSet_14BDrafter,max_tokens=16384,big_model_gpus=0,small_model_gpus=1,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/NewModel_e2egrpo_8b_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>"  

# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason_perf=True    ,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-E2EGRPO-OpenR1_Math_SpecR_GRPO_Mini-MiniSet,max_tokens=16384,big_model_gpus=0,small_model_gpus=1,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/NewModel_e2egrpo_8b_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>"  


# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason_perf=True    ,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-E2EGRPO-OpenR1_Math_SpecR_GRPO_Mini-MiniSet,max_tokens=16384,big_model_gpus=0,small_model_gpus=1,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/NewModel_e2egrpo_8b_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>"  



# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason_perf=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-E2EGRPO-OpenR1_Math_SpecR_GRPO_Mini-MiniSet,max_tokens=16384,big_model_gpus=0,small_model_gpus=1,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/NewModel_e2egrpo_8b --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>"  


# 10 repetitions of AIME25 No Figures, AIME 24 No Figures

# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason_perf=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0,small_model_gpus=1|2,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#      --tasks gpqa_diamond_openai  --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_8B_8Aug --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>"  



# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason_perf=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0,small_model_gpus=1|2,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#      --tasks gpqa_diamond_openai  --limit 10 --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_8B_8Aug --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>"  


# # SpecReason
# for i in {1..10}; do
#      python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason_perf=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0,small_model_gpus=1|2,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#           --tasks aime24_nofigures,aime25_nofigures  --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_8B_8Aug --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>"  
# done

# fuser -k -9 /dev/nvidia*

# # 1.5B only
# for i in {1..10}; do
#      python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,small_model_only=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0,small_model_gpus=1|2,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#           --tasks aime24_nofigures,aime25_nofigures  --batch_size auto --apply_chat_template  --output_path log_traces/SMALL_ONLY_8Aug --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>"  
# done

# fuser -k -9 /dev/nvidia*
# # 8B only
# for i in {1..10}; do
#      python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,big_model_only=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=1|2,small_model_gpus=0,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#           --tasks aime24_nofigures,aime25_nofigures  --batch_size auto --apply_chat_template  --output_path log_traces/BIG_ONLY_8Aug --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>"  
# done

# fuser -k -9 /dev/nvidia*

# # SpecReason
# for i in {1..10}; do
#      python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason_perf=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=1|2,small_model_gpus=0,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#           --tasks aime24_nofigures,aime25_nofigures  --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_14B_8Aug --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>"  
# done

# fuser -k -9 /dev/nvidia*

# # SpecReason
# for i in {1..10}; do
#      python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason_perf=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=1|2,small_model_gpus=0,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#           --tasks aime24_nofigures,aime25_nofigures  --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_32B_8Aug --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>"  
# done


# fuser -k -9 /dev/nvidia*

# fuser -k -9 /dev/nvidia*     
### Debug mode to start performance support
# # Test to have 8B big model, 1.5B small model
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,big_model_only=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=1,small_model_gpus=0,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/DEBUGGING_SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 


# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason_perf=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=1,small_model_gpus=0,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#      --tasks aime25_nofigures  --batch_size auto --apply_chat_template  --output_path log_traces/DEBUGGING_SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,small_model_only=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0,small_model_gpus=1|2,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#      --tasks aime25_nofigures  --batch_size auto --apply_chat_template  --output_path log_traces/DEBUGGING_SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 



# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/DEBUGGING_SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model_port=8002,small_model_port=8004,port=5002"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/DEBUGGING_SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 


# Note, it is best to average / median results over several generations :) 
# Most pass@1 do 16 generations, but we rely on simply repeating the experiment 5 times due to compute constraints.

# fuser -k -9 /dev/nvidia*
# # Test to have the finetuned 1.5B model [ONLY] profiled.
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,small_model_only=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/POST_TRAIN_15b --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,small_model_only=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/POST_TRAIN_15b --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,small_model_only=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/POST_TRAIN_15b --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,small_model_only=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/POST_TRAIN_15b --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,small_model_only=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/POST_TRAIN_15b --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 


# fuser -k -9 /dev/nvidia*
# # Test to have 32B big model, 1.5B small model
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_32B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_32B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_32B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_32B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_32B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
     
# # Test to have 14B big model, 1.5B small model
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_14B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_14B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_14B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_14B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_14B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# # fuser -k -9 /dev/nvidia*     

# # # Test to have 8B big model, 1.5B small model
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_8B --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# fuser -k -9 /dev/nvidia*     

# # # Test to have 7B big model, 1.5B small model
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_7B_Qwen --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_7B_Qwen --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_7B_Qwen --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_7B_Qwen --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,spec_reason=True,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner,max_tokens=16384,big_model_gpus=0|1,small_model_gpus=2,pretrained=meta-llama/Llama-2-7b-chat-hf"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SPECR_7B_Qwen --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
