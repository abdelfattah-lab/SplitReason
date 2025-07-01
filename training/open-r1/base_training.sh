### SFT Training Script

ACCELERATE_LOG_LEVEL=info MASTER_PORT=29501 accelerate launch --main_process_port 29502 \
    --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/sft/config_demo.yaml \
    --wandb_project SpeculativeReasoning --run_name DeepSeek-R1-Distill-Qwen-1.5B-SpeculativeReasoning

### GRPO Training Scripts

# Background Runner for GRPO service
python3 ./spec_service.py \
  --spec_reason_perf \
  --big_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --small_model akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpeculativeReasoner \
  --max_tokens 16384 \
  --big_model_gpus 0 \
  --small_model_gpus 1 \
  --big_model_port 8002 \
  --small_model_port 8004 \
  --port 5002

# Foreground Training for GRPO
CUDA_VISIBLE_DEVICES=2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 2 \
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
    --wandb_project SpeculativeReasoning --run_name DeepSeek-R1-Distill-Qwen-1.5B-E2EGRPO-OpenR1_Math_SpecR_GRPO_Mini-H100

###### 32B Drafter Model Training Run

# Background Runner for GRPO service
python3 ./spec_service.py \
  --spec_reason_perf \
  --big_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --small_model akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpeculativeReasoner \
  --max_tokens 16384 \
  --big_model_gpus "0,1" \
  --small_model_gpus 2 \
  --big_model_port 8002 \
  --small_model_port 8004 \
  --port 5002

# Foreground Training for GRPO
CUDA_VISIBLE_DEVICES=3,4 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 2 \
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo_32b.yaml \
    --wandb_project SpeculativeReasoning --run_name DeepSeek-R1-Distill-Qwen-1.5B-E2EGRPO-OpenR1_Math_SpecR_GRPO_Mini-H100-32B-Drafter


###### 14B Drafter Model Training Run

# Background Runner for GRPO service
python3 ./spec_service.py \
  --spec_reason_perf \
  --big_model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --small_model akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpeculativeReasoner \
  --max_tokens 16384 \
  --big_model_gpus "5,6" \
  --small_model_gpus "7" \
  --big_model_port 8002 \
  --small_model_port 8004 \
  --port 5002



#### CURRENT RUN
# akhauriyash/E2EGRPO_bm14B_32Gen_8GAcc_2K_2xAccF
# akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpeculativeReasoner --> original model, not used when resuming.
python3 ./spec_service.py \
  --spec_reason_perf \
  --big_model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --small_model akhauriyash/E2EGRPO_bm14B_32Gen_8GAcc_2K_2xAccF \
  --max_tokens 16384 \
  --big_model_gpus "4" \
  --small_model_gpus "5" \
  --big_model_port 8002 \
  --small_model_port 8004 \
  --port 5002
  
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120000
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=216000
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
  accelerate launch --main_process_port 29507 --config_file recipes/accelerate_configs/zero3.yaml --num_processes 4 \
  src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo_14b.yaml \
  --wandb_project SpeculativeReasoning --run_name E2EGRPO_bm14B_32Gen_8GAcc_2K_2xAccF
    

#### CURRENT RUN

# # Foreground Training for GRPO
# sleep 240; 
# TORCH_NCCL_ENABLE_MONITORING=0 CUDA_VISIBLE_DEVICES=1,2 ACCELERATE_LOG_LEVEL=info \
#     accelerate launch --main_process_port 29507 --config_file recipes/accelerate_configs/zero3.yaml --num_processes 2 \
#     src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo_14b.yaml \
#     --wandb_project SpeculativeReasoning --run_name E2EGRPO_bm14B_64Gen_4K_2xAcc
  
# sleep 240; CUDA_VISIBLE_DEVICES=4,5 ACCELERATE_LOG_LEVEL=info \
#     accelerate launch --main_process_port 29507 --config_file recipes/accelerate_configs/zero3.yaml --num_processes 2 \
#     src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo_14b.yaml \
#     --wandb_project SpeculativeReasoning --run_name 1.5B_E2EGRPO_14B_128Gen_4K_2xAcc_v2

# Foreground Training for GRPO
# CUDA_VISIBLE_DEVICES=2,3 ACCELERATE_LOG_LEVEL=info \
#     accelerate launch --main_process_port 29507 --config_file recipes/accelerate_configs/zero2.yaml --num_processes 2 \
#     src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo_14b.yaml \
#     --wandb_project SpeculativeReasoning --run_name 1.5B_E2EGRPO_14B_128Gen_4K_2xAcc


# CUDA_VISIBLE_DEVICES=2,3,4,5 ACCELERATE_LOG_LEVEL=info \
#     accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 \
#     src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
#     --wandb_project SpeculativeReasoning --run_name DeepSeek-R1-Distill-Qwen-1.5B-E2EGRPO-OpenR1-220K




        # ##### TRAINING SFT SCRIPT #####

        # ACCELERATE_LOG_LEVEL=info MASTER_PORT=29501 accelerate launch --main_process_port 29502 \
        #     --config_file recipes/accelerate_configs/zero3.yaml \
        #     src/open_r1/sft.py \
        #     --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/sft/config_demo.yaml \
        #     --wandb_project SpeculativeReasoning --run_name DeepSeek-R1-Distill-Qwen-1.5B-SpeculativeReasoning

        # ##### GRPO SCRIPT #####

        # CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpeculativeReasoner

        # CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
        #     accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 7 \
        #     src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
        #     --wandb_project SpeculativeReasoning --run_name DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner



        # ##### SplitGRPO SCRIPT #####

        # # CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpeculativeReasoner

        # # python3 ./spec_service.py \
        # #   --spec_reason_perf \
        # #   --big_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        # #   --small_model akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpeculativeReasoner \
        # #   --max_tokens 16384 \
        # #   --big_model_gpus 0 \
        # #   --small_model_gpus 1 \
        # #   --big_model_port 8002 \
        # #   --small_model_port 8004 \
        # #   --port 5002

        # # CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
        # #     accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 7 \
        # #     src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
        # #     --wandb_project SpeculativeReasoning --run_name DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner


