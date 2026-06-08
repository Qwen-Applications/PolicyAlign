import json
import argparse
import pandas as pd
import common
import os
from local_sampler import LocalSampler
from config import MODEL_CONFIG

import pytz
from datetime import datetime

get_time = lambda: datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime('%y%m%d-%H%M')


OPCD_GEN_DEFAULTS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": 8192,
    "context_length": 16384,
    "gpu_memory_utilization": 0.90,
}


def build_model_list(args, model_config):
    model_list = {}

    if args.model_path is not None:
        model_list[args.model] = LocalSampler(
            model=args.model_path,
            system_message=args.system_prompt,
            ngpu=args.ngpu,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.random_seed,
            top_p=args.top_p,
            top_k=args.top_k,
            context_length=args.context_length,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        return model_list

    for model_name, config in model_config.items():
        print(model_name, config)
        model_list[model_name] = LocalSampler(
            model=config["model_path"],
            system_message=(
            args.system_prompt
            if args.system_prompt is not None
            else config.get("system_prompt", None)
        ),
            ngpu=args.ngpu,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.random_seed,
            top_p=args.top_p,
            top_k=args.top_k,
            context_length=args.context_length,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    return model_list


def file_path_gen(args, eval_name, model_name, debug_suffix, file_type="json", output_dir="result"):
    file_dir = f'{output_dir}/{model_name.split("/")[-1]}'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = f'{file_dir}/{eval_name}{debug_suffix}'
    for param, default_val in [
        ("max_tokens", OPCD_GEN_DEFAULTS["max_tokens"]),
        ("temperature", OPCD_GEN_DEFAULTS["temperature"]),
        ("top_p", OPCD_GEN_DEFAULTS["top_p"]),
        ("top_k", OPCD_GEN_DEFAULTS["top_k"]),
        ("context_length", OPCD_GEN_DEFAULTS["context_length"]),
    ]:
        if getattr(args, param) != default_val:
            file_path += f'_{param}-{getattr(args, param)}'
    timestamp = get_time()
    file_path += f'_{timestamp}.{file_type}'
    return file_path


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using vLLM-backed local models or API graders."
    )
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, default="my_opcd_model", help="Model alias used in result naming")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the local HuggingFace model directory")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt")
    parser.add_argument("--debug", type=lambda x: (str(x).lower() == 'true'), help="Run in debug mode", default=False)
    parser.add_argument("--examples", type=int, help="Number of examples to use (overrides default)")
    parser.add_argument("--api_model", type=bool, help="The sampler is an API model or not", default=False)
    parser.add_argument("--eval_mode", type=str, help="Select the benchmark by name", default="gpqa")
    parser.add_argument("--temperature", type=float, default=OPCD_GEN_DEFAULTS["temperature"])
    parser.add_argument("--top_p", type=float, default=OPCD_GEN_DEFAULTS["top_p"])
    parser.add_argument("--top_k", type=int, default=OPCD_GEN_DEFAULTS["top_k"])
    parser.add_argument("--max_tokens", type=int, default=OPCD_GEN_DEFAULTS["max_tokens"], help="Maximum generated tokens")
    parser.add_argument("--context_length", type=int, default=OPCD_GEN_DEFAULTS["context_length"], help="vLLM max_model_len")
    parser.add_argument("--gpu_memory_utilization", type=float, default=OPCD_GEN_DEFAULTS["gpu_memory_utilization"])
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--ngpu", type=int, help="Number of GPUs / tensor parallel size", default=1)
    args = parser.parse_args()

    MODEL_LIST = build_model_list(args, MODEL_CONFIG)

    if args.list_models:
        print("Available models:")
        for model_name in MODEL_LIST.keys():
            print(f" - {model_name}")
        return

    if args.model not in MODEL_LIST:
        print(f"Error: Model '{args.model}' not found.")
        return
    models = {args.model: MODEL_LIST[args.model]}

    def get_evals(eval_name, debug_mode):
        num_examples = args.examples if args.examples is not None else (10 if debug_mode else None)
        match eval_name:
            case "mmlu_pro":
                from mmlu_pro_eval import MMLUProEval
                return MMLUProEval(num_examples=10 if debug_mode else num_examples, random_seed=args.random_seed)
            case "math":
                from math_eval import MathEval
                return MathEval(
                    num_examples=num_examples,
                    n_repeats=1,
                    random_seed=args.random_seed,
                    split="math_500_test",
                )
            case "gpqa":
                from gpqa_eval import GPQAEval
                return GPQAEval(
                    n_repeats=1,
                    num_examples=num_examples,
                    random_seed=args.random_seed,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {args.eval_mode: get_evals(args.eval_mode, args.debug)}
    debug_suffix = "_DEBUG" if args.debug else ""
    mergekey2resultpath = {}

    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            file_stem = f'{eval_name}_{model_name.split("/")[-1]}'
            gen_filename = file_path_gen(args, eval_name, model_name, debug_suffix + "_Gen", "json")
            result = eval_obj(sampler, gen_filename)
            report_filename = file_path_gen(args, eval_name, model_name, debug_suffix, "html")
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w", encoding="utf-8") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = file_path_gen(args, eval_name, model_name, debug_suffix, "json")
            with open(result_filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(metrics, indent=2, ensure_ascii=False))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename

    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r", encoding="utf-8"))
        except Exception as e:
            print(e, result_filename)
            continue
        result_val = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append({"eval_name": eval_name, "model_name": model_name, "metric": result_val})
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(index=["model_name"], columns="eval_name")

    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
