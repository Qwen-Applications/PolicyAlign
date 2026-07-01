<div align="center">

## PolicyAlign: Direct Policy-Based Safety Alignment for Large Language Models

<!-- Badges -->
<a><img 
     src="https://img.shields.io/badge/Qwen-Applications-4433FF?style=for-the-badge&logo=data:image/png;base64,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" 
     alt="Qwen"></a>
<a href="https://arxiv.org/abs/2606.25442"><img src="https://img.shields.io/badge/arXiv-2606.25442-b31b1b.svg?style=for-the-badge" alt="arXiv"></a>
<a href="https://github.com/Qwen-Applications/PolicyAlign"><img src="https://img.shields.io/badge/Github-Code-black?style=for-the-badge&logo=github" alt="Github"></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge" alt="License"></a>

<p align="center">
  <i><b> <img src="https://img.alicdn.com/imgextra/i2/O1CN01FPcQDy1WTPjPX6IH9_!!6000000002789-2-tps-96-96.png" width="16px"  style="vertical-align: middle;"> Qwen Large Model Application Team, Alibaba</b></i>
</p>

</div>

### 📖 Overview

Safety alignment of LLMs typically depends on high-quality supervision data, such as safe demonstrations or preference pairs. However, in real-world deployment, emerging safety requirements are often specified as **natural-language policies**, while corresponding supervision data may be costly, delayed, or unavailable.

**PolicyAlign** is a simple yet effective framework for **directly aligning LLMs with safety policies**. Given a safety policy, PolicyAlign:

1. **Synthesizes policy-violating instructions** to construct targeted training data
2. **Performs on-policy self-distillation** to internalize policy-guided behavior
3. **Applies Policy-Sensitive Filtering** to select instructions where the policy induces the largest behavioral shift, improving training stability and data efficiency

PolicyAlign consistently improves safety while maintaining low over-refusal and preserving general capabilities. It also generalizes across multiple safety domains:

| Domain | System Prompt | Description |
|--------|--------------|-------------|
| **LLM Safety** | [`system_prompts/llm_safety.txt`](system_prompts/llm_safety.txt) | General AI safety covering hate speech, violence, privacy, misinformation, etc. |
| **Medical Safety** | [`system_prompts/medical_safety.txt`](system_prompts/medical_safety.txt) | Medical ethics, patient privacy, evidence-based practice |
| **Legal Safety** | [`system_prompts/law_safety.txt`](system_prompts/law_safety.txt) | Legal compliance and ethical legal practice |
| **Financial Safety** | [`system_prompts/finance_safety.txt`](system_prompts/finance_safety.txt) | Financial regulations and responsible advisory |

---

### 🔧 Requirements

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

---

### 🚀 Getting Started

#### 1. Launch the Guard Model

Start the Qwen3Guard-Gen-8B safety review service:

```bash
bash guard_server.sh
```

#### 2. Run Training

```bash
bash scripts/sys_consolidate.sh \
  --model Qwen2.5-7B-Instruct \
  --exp_name align-qwen7b \
  --nnodes 1 \
  --rollout_n 1 \
  --kl_loss_type full \
  --kl_topk 256 \
  --actor_lr 5e-6 \
  --max_response_length 1024 \
  --experience_max_length 1024 \
  --system_prompt_type safety \
  --exp_path system_prompts/llm_safety.txt \
  --total_training_steps 100 \
  --save_freq 10 \
  --teacher_mode ema \
  --teacher_ema_decay 0.99
```

#### 3. Run Evaluation

**Safety benchmarks** (StrongREJECT, Advbench, Wildjailbreak, Fortress):

```bash
bash scripts/sys_eval.sh \
  --model Qwen2.5-7B-Instruct \
  --exp_name eval-qwen7b \
  --nnodes 1 \
  --ckpt 0 \
  --prompt_version v4 \
  --use_bsl true \
  --eval_prepend_experience false \
  --exp_path system_prompts/llm_safety.txt \
  --max_response_length 1024 \
  --experience_max_length 2048 \
  --system_prompt_type safety \
  --system_prompt_version v1
```

**XSTest**:

```bash
python data/xstest_eval.py --file_path /path/xstest_eval
```

**General capabilities** (MMLU-Pro, GPQA-Diamond, MATH500):

```bash
cd general_eval
python simple_evals.py \
  --model Qwen2.5-7B-Instruct \
  --model_path /path/model \
  --eval_mode mmlu_pro  # or gpqa / math
```

---

### 📁 Project Structure

```
PolicyAlign/
├── scripts/                    # Training & evaluation scripts
│   ├── sys_consolidate.sh      #   PPO alignment training
│   └── sys_eval.sh             #   Safety & capability evaluation
├── system_prompts/             # Domain-specific safety policies
│   ├── llm_safety.txt          #   General LLM safety rules
│   ├── medical_safety.txt      #   Medical domain policies
│   ├── law_safety.txt          #   Legal domain policies
│   └── finance_safety.txt      #   Financial domain policies
├── general_eval/               # General capability evaluation
│   ├── simple_evals.py         #   Evaluation entry point
│   ├── mmlu_pro_eval.py        #   MMLU-Pro benchmark
│   ├── gpqa_eval.py            #   GPQA-Diamond benchmark
│   └── math_eval.py            #   MATH-500 benchmark
├── tools/                      # Utility scripts
│   ├── prepare_data.py         #   Data preparation
│   ├── merge_model2hf.py       #   Checkpoint merging
│   └── make_exp_list.py        #   Experiment management
├── verl/                       # Core RL framework (modified verl)
├── guard_server.sh             # Guard model service launcher
└── README.md                   # This file
```

---

### 🙏 Acknowledgements

This codebase is built upon [verl](https://github.com/volcengine/verl) and [OPCD](https://github.com/microsoft/LMOps/tree/main/opcd). We thank all teams for their excellent open-source contributions.

---

### 📜 Citation

If you find our work useful, please consider citing:

```bibtex
@article{wu2026policyalign,
  title={PolicyAlign: Direct Policy-Based Safety Alignment for Large Language Models},
  author={Wu, Chang and Fang, Junfeng and Jiang, Houcheng and Tang, Kai and Cheng, Pengyu and Jiang, Xiaoxi and Jiang, Guanjun and Wang, Xiang},
  journal={arXiv preprint arXiv:2606.25442},
  year={2026}
}
```
