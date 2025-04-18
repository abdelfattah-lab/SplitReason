

# ACCELERATE_LOG_LEVEL=info MASTER_PORT=29501 accelerate launch --main_process_port 29502 --config_file recipes/accelerate_configs/zero3.yaml \
#     src/open_r1/sft.py \
#     --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/sft/config_demo.yaml --wandb_project SpeculativeReasoning --run_name DeepSeek-R1-Distill-Qwen-1.5B-SpeculativeReasoning

CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpeculativeReasoner

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 7 \
    src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml --wandb_project SpeculativeReasoning --run_name DeepSeek-R1-Distill-Qwen-1.5B-GRPO-SpeculativeReasoner



    # CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model akhauriyash/DeepSeek-R1-Distill-Qwen-1.5B-SpecReasoner_SFT_14k

    # CUDA_VISIBLE_DEVICES=1,2,3 ACCELERATE_LOG_LEVEL=info     accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 3     src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml --wandb_project SpeculativeReasoning --run_name DeepSeek-R1-Distill-Qwen-1.5B-SpecReasoner_SFT_GRPO_14k