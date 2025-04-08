

ACCELERATE_LOG_LEVEL=info MASTER_PORT=29504 accelerate launch --main_process_port 29503 --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/sft/config_demo.yaml




# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
#     src/open_r1/grpo.py \
#     --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml

# CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# CUDA_VISIBLE_DEVICES=1,2,3 ACCELERATE_LOG_LEVEL=info \
#     accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 3 \
#     src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml

