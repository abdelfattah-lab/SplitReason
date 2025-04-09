#!/usr/bin/env bash

export $(grep -v '^#' .env | xargs)

export VLLM_WORKER_MULTIPROC_METHOD='spawn'
export PROCESSOR=gpt-4o-mini
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=logging_config.json
##### RUN THIS BEFORE STARTING NEW JOBS #####
# fuser -k -9 /dev/nvidia*
# sleep 20
##### RUN THIS BEFORE STARTING NEW JOBS #####


# SPECULATIVE BIGMODEL REASONING BASE SMALL MODEL ACCURACY PRESERVATION TESTS
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpecReasoner,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SR_15b_only_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpecReasoner,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SR_15b_only_v1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpecReasoner,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SR_15b_only_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpecReasoner,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SR_15b_only_v3 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpecReasoner,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/SR_15b_only_v4 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 



# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8007,big_model_gpus=1|2,small_model=akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpecReasoner,small_model_port=8003,small_model_gpus=0,max_tokens=16384,switch_chunk=16,spec_reason=True,switch_ratio=1"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/specreason_test0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python test_spec.py --test_logging --big_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --big_model_gpus 1,2 --small_model_gpus 0 --small_model akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpecReasoner --spec_reason --small_model_port 8003 --big_model_port 8007



### ALL TESTS BELOW THIS ARE NOT USED ###
# ###### Basic testing sample ######
# python test_spec.py --test_logging --big_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --big_model_gpus 0 --small_model_gpus 1 --small_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --logprob_subselect --sgen 512 --stok 16 --sdecay 2 --ltok 32
# python test_spec.py --test_logging --big_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --big_model_gpus 1|2 --small_model_gpus 0 --small_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --random_switch --switch_ratio 4 --switch_chunk 32

# # 1b model multiple repeats
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/15b_only_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20

#  8-switch 128 tok

# fuser -k -9 /dev/nvidia*
# sleep 20

# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=128,random_switch=True,switch_ratio=8"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_8_sc_128_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=128,random_switch=True,switch_ratio=8"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_8_sc_128_v1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=128,random_switch=True,switch_ratio=8"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_8_sc_128_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=128,random_switch=True,switch_ratio=8"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_8_sc_128_v3 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=128,random_switch=True,switch_ratio=8"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_8_sc_128_v4 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20

    
# #  4-switch 64 tok
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=4"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_4_sc_64_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=4"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_4_sc_64_v1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=4"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_4_sc_64_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=4"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_4_sc_64_v3 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=4"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_4_sc_64_v4 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20

# # Switching for 64 tokens every time
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=1"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_64_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=1"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_64_v1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=1"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_64_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=1"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_64_v3 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=1"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_64_v4 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20

# # Switching for 16 tokens every time
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=16,random_switch=True,switch_ratio=1"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_16_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=16,random_switch=True,switch_ratio=1"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_16_v1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=16,random_switch=True,switch_ratio=1"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_16_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=16,random_switch=True,switch_ratio=1"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_16_v3 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=16,random_switch=True,switch_ratio=1"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_16_v4 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20

# # 1b model multiple repeats
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/15b_only_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/15b_only_v1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/15b_only_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/15b_only_v3 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/15b_only_v4 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20


# # 32b model multiple repeats
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,big_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,big_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_v1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,big_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,big_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_v3 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20
     
# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,big_model_only=True,sequential_scale=0"  \
#      --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_v4 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# fuser -k -9 /dev/nvidia*
# sleep 20

