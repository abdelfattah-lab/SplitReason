#!/usr/bin/env bash

export $(grep -v '^#' .env | xargs)

export VLLM_WORKER_MULTIPROC_METHOD='fork'
export PROCESSOR=gpt-4o-mini
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=logging_config.json
export PYTHONPATH="${PYTHONPATH}:${PWD}"
export ZMQ_BIND_ADDRESS='127.0.0.1'
export CUDA_LAUNCH_BLOCKING=1
export OPENAI_API_KEY=sk-f1NsPRPJmnjiOWU_A8--7Q48QK4q4nETb0mAYTr51iT3BlbkFJph5v94eagNhCriCnJw-FDtdQTJNIxsBYQUbMHp_IsA


### Spe


#### Speculative Decoding

python -m lm_eval --model vllm_speculative --model_args "service_script_path=./../spec_service.py,pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,spec_decode=True"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/spec_decode --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

#### Speculative Decoding through big_model only

python -m lm_eval --model vllm_speculative --model_args "service_script_path=./../spec_service.py,pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,max_tokens=16384,big_model_only=True"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/spec_decode --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 



##### RUN THIS BEFORE STARTING NEW JOBS #####
# fuser -k -9 /dev/nvidia*
##### RUN THIS BEFORE STARTING NEW JOBS #####

# ###### Basic testing sample ######
# python test_spec.py --test_logging --big_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --big_model_gpus 0 --small_model_gpus 1 --small_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --logprob_subselect --sgen 512 --stok 16 --sdecay 2 --ltok 32
# python test_spec.py --test_logging --big_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --big_model_gpus 1|2 --small_model_gpus 0 --small_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --random_switch --switch_ratio 4 --switch_chunk 32


python -m lm_eval --model vllm_speculative --model_args "service_script_path=./../spec_service.py,pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=1,random_switch=True,switch_ratio=128"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_8_sc_128_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 


#  8-switch 128 tok
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=128,random_switch=True,switch_ratio=8"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_8_sc_128_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=128,random_switch=True,switch_ratio=8"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_8_sc_128_v1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=128,random_switch=True,switch_ratio=8"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_8_sc_128_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=128,random_switch=True,switch_ratio=8"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_8_sc_128_v3 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=128,random_switch=True,switch_ratio=8"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_8_sc_128_v4 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

    
#  4-switch 64 tok
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=4"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_4_sc_64_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=4"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_4_sc_64_v1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=4"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_4_sc_64_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=4"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_4_sc_64_v3 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=4"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_4_sc_64_v4 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# Switching for 64 tokens every time
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=1"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_64_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=1"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_64_v1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=1"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_64_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=1"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_64_v3 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=64,random_switch=True,switch_ratio=1"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_64_v4 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# Switching for 16 tokens every time
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=16,random_switch=True,switch_ratio=1"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_16_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=16,random_switch=True,switch_ratio=1"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_16_v1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=16,random_switch=True,switch_ratio=1"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_16_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=16,random_switch=True,switch_ratio=1"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_16_v3 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,switch_chunk=16,random_switch=True,switch_ratio=1"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/rswitch_sw_1_sc_16_v4 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# 1b model multiple repeats
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/15b_only_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/15b_only_v1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/15b_only_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/15b_only_v3 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,small_model_only=True,sequential_scale=0"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/15b_only_v4 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 


# 32b model multiple repeats
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,big_model_only=True,sequential_scale=0"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_v0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,big_model_only=True,sequential_scale=0"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_v1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,big_model_only=True,sequential_scale=0"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_v2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,big_model_only=True,sequential_scale=0"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_v3 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
     
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,big_model_only=True,sequential_scale=0"  \
     --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_v4 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 



# python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=32,sgen=8,sdecay=2,ltok=512,lbound=8,logprob_subselect=True"   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_14b_1_5b_512_64_2_32_8 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
python -m lm_eval --model vllm_speculative --model_args "service_script_path=./spec_service.py,pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=0|1,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=2,max_tokens=16384,stok=32,sgen=8,sdecay=2,ltok=512,lbound=8,logprob_subselect=True"   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/random_switch --random_switch --switch_ratio 32 --switch_chunk 1 --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# ###### TESTING WITH Q7B _ Q1.5B ############ TESTING WITH Q7B _ Q1.5B ############ TESTING WITH Q7B _ Q1.5B ############ TESTING WITH Q7B _ Q1.5B ############ TESTING WITH Q7B _ Q1.5B ######
# # # LogProb Subselect Qwen1.5 512_16_2_32_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,big_model_port=8000,big_model_gpus=0,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=1,max_tokens=16384,stok=512,sgen=16,sdecay=2,ltok=512,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_7b_15b_512_32_2_32_2_new --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 


# # # LogProb Subselect Qwen1.5 512_16_2_32_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=0|1,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=2,max_tokens=16384,stok=512,sgen=16,sdecay=2,ltok=32,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_512_32_2_32_2_new --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 




# ###### TESTING WITH Q32B _ Q1.5B ############ TESTING WITH Q32B _ Q1.5B ############ TESTING WITH Q32B _ Q1.5B ############ TESTING WITH Q32B _ Q1.5B ############ TESTING WITH Q32B _ Q1.5B ######
# # # LogProb Subselect Qwen1.5 512_16_2_32_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_512_16_2_32_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# # # LogProb Subselect Qwen1.5 512_16_2_64_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_512_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# Thinking only
# Pretrained argument is needed to apply the chat template successfully to the question.
# max_gen_toks is needed to actually have questions in the prompts for some reason.
python -m lm_eval --model vllm_thinkonly \
  --model_args "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,big_model_port=8000,big_model_gpus=0,max_tokens=16384,think_tokens=256,num_rounds=3,debug_prompts=true" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template --output_path log_traces/think_summarize --log_samples --gen_kwargs "max_gen_toks=16384" --limit 1


# Summarization
python -m lm_eval --model vllm_summarize \
  --model_args "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,big_model_port=8000,big_model_gpus=0,max_tokens=16384,think_tokens=256,summary_tokens=256,num_rounds=3,debug_prompts=true" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template --output_path log_traces/think_summarize --log_samples --gen_kwargs "max_gen_toks=16384" --limit 1


# Test the new think-summarize model with the 32B model
time python -m lm_eval --model vllm_summarize \
  --model_args "pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,max_tokens=16384,think_tokens=200,num_rounds=3,debug_prompts=true" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template --output_path log_traces/think_summarize --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# # Logprob subselect on 1.5B Qwen model with 14B guide
# time python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=16,sdecay=2,ltok=32,lbound=2,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template --output_path log_traces/logprob_subset --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  

# # # LogProb Subselect Qwen1.5 256_16_2_64_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_256_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  

# # # LogProb Subselect Qwen1.5 256_32_2_64_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_256_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  

# # # LogProb Subselect Qwen1.5 256_64_2_64_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_256_64_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# ###### TESTING WITH Q14B _ Q1.5B ############ TESTING WITH Q14B _ Q1.5B ############ TESTING WITH Q14B _ Q1.5B ############ TESTING WITH Q14B _ Q1.5B ############ TESTING WITH Q14B _ Q1.5B ######
# # # LogProb Subselect Qwen1.5 512_16_2_32_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_14b_15b_512_16_2_32_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  
# # # LogProb Subselect Qwen1.5 512_16_2_64_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_14b_15b_512_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  

# # # LogProb Subselect Qwen1.5 256_16_2_64_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_14b_15b_256_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  

# # # LogProb Subselect Qwen1.5 256_32_2_64_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_14b_15b_256_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  

# # # LogProb Subselect Qwen1.5 256_64_2_64_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_14b_15b_256_64_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 


# ###### TESTING WITH Q14 ############ TESTING WITH Q14 ############ TESTING WITH Q14 ############ TESTING WITH Q14 ############ TESTING WITH Q14 ############ TESTING WITH Q14 ######
# # # Big Model Only sequential_scale 0
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,big_model_only=True,sequential_scale=0" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/14b_only_seqscale0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # # Big Model Only sequential_scale 1
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,big_model_only=True,sequential_scale=1" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/14b_only_seqscale1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # # Big Model Only sequential_scale 2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,big_model_only=True,sequential_scale=2" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/14b_only_seqscale2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 


# ###### TESTING WITH Q32 ############ TESTING WITH Q32 ############ TESTING WITH Q32 ############ TESTING WITH Q32 ############ TESTING WITH Q32 ############ TESTING WITH Q32 ######
# # # Big Model Only sequential_scale 0
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,big_model_only=True,sequential_scale=0" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_seqscale0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # # Big Model Only sequential_scale 1
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,big_model_only=True,sequential_scale=1" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_seqscale1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # # Big Model Only sequential_scale 2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,big_model_only=True,sequential_scale=2" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_seqscale2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# ###### TESTING WITH Q1.5b ############ TESTING WITH Q1.5b ############ TESTING WITH Q1.5b ############ TESTING WITH Q1.5b ############ TESTING WITH Q1.5b ############ TESTING WITH Q1.5b ######
# # # Small Model Only sequential_scale 0
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,small_model_only=True,sequential_scale=0" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/1_5b_only_seqscale0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # # Small Model Only sequential_scale 1
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,small_model_only=True,sequential_scale=1" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/1_5b_only_seqscale1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # # Small Model Only sequential_scale 2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,small_model_only=True,sequential_scale=2" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/1_5b_only_seqscale2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# ###### TESTING WITH Q32B _ Q7B ############ TESTING WITH Q32B _ Q7B ############ TESTING WITH Q32B _ Q7B ############ TESTING WITH Q32B _ Q7B ############ TESTING WITH Q32B _ Q7B ######

# # # LogProb Subselect Qwen1.5 512_16_2_32_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_512_16_2_32_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  
# # # LogProb Subselect Qwen1.5 512_16_2_64_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_512_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# # # LogProb Subselect Qwen1.5 256_16_2_64_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_256_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# # # LogProb Subselect Qwen1.5 256_32_2_64_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_256_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  
# # # LogProb Subselect Qwen1.5 256_64_2_64_2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_256_64_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# ###### TESTING WITH Q7b ############ TESTING WITH Q7b ############ TESTING WITH Q7b ############ TESTING WITH Q7b ############ TESTING WITH Q7b ############ TESTING WITH Q7b ############ TESTING WITH Q7b ######

# # # Small Model Only sequential_scale 0
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,small_model_only=True,sequential_scale=0" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/7b_only_seqscale0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # # Small Model Only sequential_scale 1
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,small_model_only=True,sequential_scale=1" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/7b_only_seqscale1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # # Small Model Only sequential_scale 2
# python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,small_model_only=True,sequential_scale=2" \
#   --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/7b_only_seqscale2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 














# time python -m lm_eval --model vllm_speculative \
# #   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=0|1,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=1,thinking_n_ignore=0,drafting_n=0,full_rewrite=True,bloat_tokens=0,max_tokens=16384,terminate_on_exit=True" \
# #     --tasks aime24_nofigures \
# #     --batch_size auto --apply_chat_template \
# #      --output_path log_traces/basic --log_samples --gen_kwargs max_gen_toks=2048
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=0|1,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=1,thinking_n_ignore=0,drafting_n=0,full_rewrite=True,bloat_tokens=0,max_tokens=16384,terminate_on_exit=True" \
#     --tasks aime24_nofigures \
#     --batch_size auto --apply_chat_template \
#     --output_path log_traces/basic --log_samples --gen_kwargs max_gen_toks=2048
time python -m lm_eval --model vllm_speculative \
 --model_args "service_script_path=./../spec_service.py,pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,big_model_port=8000,big_model_gpus=0,thinking_n_ignore=0,drafting_n=0,full_rewrite=True,bloat_tokens=0,max_tokens=16384,terminate_on_exit=True" \
    --tasks aime24_nofigures \
    --batch_size auto --apply_chat_template \
    --output_path log_traces/basic --log_samples --gen_kwargs max_gen_toks=2048

time python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./../spec_service.py,pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,big_model_port=8000,big_model_gpus=0,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=1,thinking_n_ignore=0,drafting_n=0,full_rewrite=True,bloat_tokens=0,max_tokens=16384,terminate_on_exit=True" \
    --tasks aime24_nofigures \
    --batch_size auto --apply_chat_template \
    --output_path log_traces/basic --log_samples --gen_kwargs max_gen_toks=2048

# time python -m lm_eval --model vllm_speculative \
# #   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=0|1,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=1,thinking_n_ignore=1,drafting_n=0,full_rewrite=True,bloat_tokens=0,max_tokens=16384,terminate_on_exit=True" \
# #     --tasks aime24_nofigures \
# #     --batch_size auto --apply_chat_template \
# #      --output_path log_traces/ignoreonce --log_samples --gen_kwargs max_gen_toks=2048

# time python -m lm_eval --model vllm_speculative \
# #   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=0|1,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=1,thinking_n_ignore=1,drafting_n=1,full_rewrite=True,bloat_tokens=0,max_tokens=16384,terminate_on_exit=True" \
# #     --tasks aime24_nofigures \
# #     --batch_size auto --apply_chat_template \
# #      --output_path log_traces/ignoreonce_redraftonce --log_samples --gen_kwargs max_gen_toks=2048

# time python -m lm_eval --model vllm_speculative \
# #   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=0|1,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=1,thinking_n_ignore=2,drafting_n=0,full_rewrite=True,bloat_tokens=0,max_tokens=16384,terminate_on_exit=True" \
# #     --tasks aime24_nofigures \
# #     --batch_size auto --apply_chat_template \
# #      --output_path log_traces/ignoretwice --log_samples --gen_kwargs max_gen_toks=2048