# PolicyAlign

## Requirements

You can create the environment for PolicyAlign by running the following command in order:

```bash
conda create -n policyalign python=3.10
conda activate policyalign
pip install vllm==0.8.5
pip install -e ./verl --no-deps 
pip install torchdata textarena trl==0.28.0 rouge-score
pip install codetiming hydra-core peft pybind11 pylatexenc tensordict==0.6.2 wandb ray
pip install datasets==4.0.0
pip install transformers==4.52.3
pip install setuptools==68.2.2
```

## Quick Start

1. Launch the guard model:

```bash
bash guard_server.sh
```

2. Run training:

```bash
bash scripts/sys_consolidate.sh --model Qwen2.5-7B-Instruct --exp_name align-qwen7b --nnodes 1 --rollout_n 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 1024 --experience_max_length 1024 --system_prompt_type safety --exp_path system_prompts/llm_safety.txt --total_training_steps 100 --save_freq 10  --teacher_mode ema --teacher_ema_decay 0.99  
```

3. Run evaluation:

For StrongREJECT, Advbench, Wildjailbreak, Fortress:

```bash
bash scripts/sys_eval.sh --model Qwen2.5-7B-Instruct --exp_name eval-qwen7b --nnodes 1 --ckpt 0 --prompt_version v4 --use_bsl true --eval_prepend_experience false --exp_path system_prompts/llm_safety.txt --max_response_length 1024 --experience_max_length 2048 --system_prompt_type safety --system_prompt_version v1
```

For XSTest:

```bash
python data/xstest_eval.py --file_path /path/xstest_eval
```

For MMLU-Pro, GPQA-Diamond, MATH500:

```bash
cd general_eval
python simple_evals.py --model Qwen2.5-7B-Instruct --model_path /path/model --eval_mode mmlu_pro/gpqa/math
```
