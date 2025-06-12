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
CUDA_VISIBLE_DEVICES=2,3,4,5 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 4 \
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
    --wandb_project SpeculativeReasoning --run_name DeepSeek-R1-Distill-Qwen-1.5B-E2EGRPO-OpenR1-220K




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


