#!/usr/bin/env bash

export $(grep -v '^#' .env | xargs)

export VLLM_WORKER_MULTIPROC_METHOD='spawn'
export PROCESSOR=gpt-4o-mini
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=logging_config.json
##### RUN THIS BEFORE STARTING NEW JOBS #####
fuser -k -9 /dev/nvidia*
##### RUN THIS BEFORE STARTING NEW JOBS #####

###### TESTING WITH Q32B _ Q1.5B ############ TESTING WITH Q32B _ Q1.5B ############ TESTING WITH Q32B _ Q1.5B ############ TESTING WITH Q32B _ Q1.5B ############ TESTING WITH Q32B _ Q1.5B ######
# # LogProb Subselect Qwen1.5 512_16_2_32_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_512_16_2_32_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# # LogProb Subselect Qwen1.5 512_16_2_64_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_512_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  

# # LogProb Subselect Qwen1.5 256_16_2_64_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_256_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  

# # LogProb Subselect Qwen1.5 256_32_2_64_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_256_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  

# # LogProb Subselect Qwen1.5 256_64_2_64_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_256_64_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

###### TESTING WITH Q14B _ Q1.5B ############ TESTING WITH Q14B _ Q1.5B ############ TESTING WITH Q14B _ Q1.5B ############ TESTING WITH Q14B _ Q1.5B ############ TESTING WITH Q14B _ Q1.5B ######
# # LogProb Subselect Qwen1.5 512_16_2_32_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_14b_15b_512_16_2_32_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  
# # LogProb Subselect Qwen1.5 512_16_2_64_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_14b_15b_512_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  

# # LogProb Subselect Qwen1.5 256_16_2_64_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_14b_15b_256_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  

# # LogProb Subselect Qwen1.5 256_32_2_64_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_14b_15b_256_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  

# # LogProb Subselect Qwen1.5 256_64_2_64_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_14b_15b_256_64_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 


###### TESTING WITH Q14 ############ TESTING WITH Q14 ############ TESTING WITH Q14 ############ TESTING WITH Q14 ############ TESTING WITH Q14 ############ TESTING WITH Q14 ######
# # Big Model Only sequential_scale 0
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,big_model_only=True,sequential_scale=0" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/14b_only_seqscale0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # Big Model Only sequential_scale 1
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,big_model_only=True,sequential_scale=1" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/14b_only_seqscale1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # Big Model Only sequential_scale 2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,big_model_only=True,sequential_scale=2" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/14b_only_seqscale2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 


###### TESTING WITH Q32 ############ TESTING WITH Q32 ############ TESTING WITH Q32 ############ TESTING WITH Q32 ############ TESTING WITH Q32 ############ TESTING WITH Q32 ######
# # Big Model Only sequential_scale 0
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,big_model_only=True,sequential_scale=0" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_seqscale0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # Big Model Only sequential_scale 1
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,big_model_only=True,sequential_scale=1" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_seqscale1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # Big Model Only sequential_scale 2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,big_model_only=True,sequential_scale=2" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/32b_only_seqscale2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

###### TESTING WITH Q1.5b ############ TESTING WITH Q1.5b ############ TESTING WITH Q1.5b ############ TESTING WITH Q1.5b ############ TESTING WITH Q1.5b ############ TESTING WITH Q1.5b ######
# # Small Model Only sequential_scale 0
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,small_model_only=True,sequential_scale=0" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/1_5b_only_seqscale0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # Small Model Only sequential_scale 1
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,small_model_only=True,sequential_scale=1" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/1_5b_only_seqscale1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # Small Model Only sequential_scale 2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,small_model_only=True,sequential_scale=2" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/1_5b_only_seqscale2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

###### TESTING WITH Q32B _ Q7B ############ TESTING WITH Q32B _ Q7B ############ TESTING WITH Q32B _ Q7B ############ TESTING WITH Q32B _ Q7B ############ TESTING WITH Q32B _ Q7B ######

# # LogProb Subselect Qwen1.5 512_16_2_32_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_512_16_2_32_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  
# # LogProb Subselect Qwen1.5 512_16_2_64_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_512_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# # LogProb Subselect Qwen1.5 256_16_2_64_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_256_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

# # LogProb Subselect Qwen1.5 256_32_2_64_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_256_16_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
  
# # LogProb Subselect Qwen1.5 256_64_2_64_2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=256,sgen=32,sdecay=2,ltok=64,lbound=8,logprob_subselect=True" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/lprob_32b_15b_256_64_2_64_2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 

###### TESTING WITH Q7b ############ TESTING WITH Q7b ############ TESTING WITH Q7b ############ TESTING WITH Q7b ############ TESTING WITH Q7b ############ TESTING WITH Q7b ############ TESTING WITH Q7b ######

# # Small Model Only sequential_scale 0
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,small_model_only=True,sequential_scale=0" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/7b_only_seqscale0 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # Small Model Only sequential_scale 1
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,small_model_only=True,sequential_scale=1" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/7b_only_seqscale1 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 
# # Small Model Only sequential_scale 2
python -m lm_eval --model vllm_speculative \
  --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,big_model_port=8000,big_model_gpus=1|2,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=0,max_tokens=16384,stok=512,sgen=32,sdecay=2,ltok=32,lbound=8,small_model_only=True,sequential_scale=2" \
  --tasks aime24_nofigures --batch_size auto --apply_chat_template  --output_path log_traces/7b_only_seqscale2 --log_samples --gen_kwargs "max_gen_toks=16384,thinking_start=\n<think>,thinking_end=\n</think>" 














time python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=0|1,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=1,thinking_n_ignore=0,drafting_n=0,full_rewrite=True,bloat_tokens=0,max_tokens=16384,terminate_on_exit=True" \
#     --tasks aime24_nofigures \
#     --batch_size auto --apply_chat_template \
#      --output_path log_traces/basic --log_samples --gen_kwargs max_gen_toks=2048

time python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=0|1,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=1,thinking_n_ignore=1,drafting_n=0,full_rewrite=True,bloat_tokens=0,max_tokens=16384,terminate_on_exit=True" \
#     --tasks aime24_nofigures \
#     --batch_size auto --apply_chat_template \
#      --output_path log_traces/ignoreonce --log_samples --gen_kwargs max_gen_toks=2048

time python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=0|1,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=1,thinking_n_ignore=1,drafting_n=1,full_rewrite=True,bloat_tokens=0,max_tokens=16384,terminate_on_exit=True" \
#     --tasks aime24_nofigures \
#     --batch_size auto --apply_chat_template \
#      --output_path log_traces/ignoreonce_redraftonce --log_samples --gen_kwargs max_gen_toks=2048

time python -m lm_eval --model vllm_speculative \
#   --model_args "service_script_path=./spec_service.py,pretrained=meta-llama/Llama-2-7b-chat-hf,big_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,big_model_port=8000,big_model_gpus=0|1,small_model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,small_model_port=8001,small_model_gpus=1,thinking_n_ignore=2,drafting_n=0,full_rewrite=True,bloat_tokens=0,max_tokens=16384,terminate_on_exit=True" \
#     --tasks aime24_nofigures \
#     --batch_size auto --apply_chat_template \
#      --output_path log_traces/ignoretwice --log_samples --gen_kwargs max_gen_toks=2048