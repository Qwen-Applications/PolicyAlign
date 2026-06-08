MODEL_CONFIG = {

#! BON
"DeepSeek-R1-Distill-Qwen-SInternal": {
    "model_path": "",  # model path
    "n_gpu": 4,  # GPU Number
    "run_api": False,
    "dtype": "bfloat16",
    "system_prompt": False,
    "generation_config": {
        "reasoning_mode": True
    }
},

}

DEFAULT_GEN_CONFIG = {
    'system': True,
    'temperature': 0.0,
    'topp': 1,
    'topk': -1,
    'max_tokens': 8000,
    'repeat_n': 1,
}




RSM_LIST = list(MODEL_CONFIG.keys())

EVAL_DATA = [
    'strongreject',
    'wildjailbreak',
    'jbbbehaviours',
    'wildchat', 
    'xstest',
    'orbench',
    'WJ_ADVBenign',
    'fortress',
    'SRJ_JR',
    'trotter_strong',
    'trotter_advance',
    'hcot_DS',
]


# print(MODEL_CONFIG["DeepSeek-R1-Distill-Qwen-7B"])