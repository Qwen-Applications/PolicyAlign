"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874

This module also supports loading data from local JSONL files for MATH dataset.
"""

import json
import os
import pathlib
import random
from typing import Any, Literal

import common
from classes import Eval, EvalResult, SamplerBase, SingleEvalResult

try:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
except ImportError as e:
    raise ImportError(
        "math_verify is required for MathEval. Install it with e.g. "
        "`pip install math-verify[antlr4_13_2]`."
    ) from e


DEFAULT_MATH_TEST_PATH = "math-500.jsonl"

QUERY_TEMPLATE = """
{Question}

Please reason step by step, and put your final answer within \\boxed{{}}.
""".strip()


def serialize_extracted_answer(obj: Any) -> str:
    if obj is None:
        return ""
    try:
        if isinstance(obj, (list, tuple)):
            return ", ".join(serialize_extracted_answer(x) for x in obj)
        return str(obj)
    except Exception as e:
        return f"<serialization_error: {e}>"


def load_math_dataset(path: str, split: str = "math_test") -> list:
    try:
        if os.path.exists(path):
            print(f"Loading {split} dataset from {path}")
            examples = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    if "problem" in item and "answer" in item:
                        examples.append(
                            {
                                "Question": item["problem"],
                                "Answer": item["answer"],
                                "Solution": item.get("solution", ""),
                                "Subject": item.get("subject", ""),
                                "Level": item.get("level", ""),
                                "ID": item.get("id", item.get("unique_id", "")),
                            }
                        )
                    elif "Question" in item and "Answer" in item:
                        examples.append(item)

            print(f"Successfully loaded {len(examples)} examples from {split} dataset")
            return examples
        else:
            print(f"Warning: File {path} does not exist")
            return []
    except Exception as e:
        print(f"Error loading dataset from {path}: {e}")
        return []


class MathEval(Eval):
    def __init__(
        self,
        equality_checker: SamplerBase | None = None,  # kept for backward compatibility; unused
        num_examples: int | None = None,
        n_repeats: int = 16,
        split: Literal["math_test", "math_500_test"] = "math_test",
        random_seed: int | None = None,
        data_dir: str | pathlib.Path = "datasets",
    ):
        if isinstance(data_dir, str):
            data_dir = pathlib.Path(data_dir)

        examples = load_math_dataset(DEFAULT_MATH_TEST_PATH, split)

        self.rng = random.Random(random_seed)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = self.rng.sample(examples, num_examples)

        self.examples = examples * n_repeats

        # Gold answers are usually cleaner; predictions are noisier and should prioritize boxed content.
        self.gold_extraction_config = [LatexExtractionConfig(), ExprExtractionConfig()]
        self.pred_extraction_config = [
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
        ]

    def _parse_gold(self, gold_text: str):
        try:
            return parse(gold_text, extraction_config=self.gold_extraction_config)
        except Exception as e:
            print(f"[math_verify] gold parse failed: {e} | gold={gold_text!r}")
            return []

    def _parse_prediction(self, pred_text: str):
        try:
            return parse(
                pred_text,
                extraction_config=self.pred_extraction_config,
                extraction_mode="first_match",
            )
        except Exception as e:
            print(f"[math_verify] prediction parse failed: {e} | pred={pred_text!r}")
            return []

    def _score_answer(self, pred_text: str, gold_text: str) -> tuple[float, str | None, bool]:
        gold_parsed = self._parse_gold(gold_text)
        pred_parsed = self._parse_prediction(pred_text)

        extracted_answer = serialize_extracted_answer(pred_parsed) if pred_parsed else None
        answer_extracted = bool(pred_parsed)

        if not gold_parsed or not pred_parsed:
            return 0.0, extracted_answer, answer_extracted

        try:
            # Order matters: verify(gold, answer)
            score = float(verify(gold_parsed, pred_parsed))
            return score, extracted_answer, answer_extracted
        except Exception as e:
            print(
                f"[math_verify] verify failed: {e} | gold={gold_text!r} | pred={pred_text!r}"
            )
            return 0.0, extracted_answer, answer_extracted

    def __call__(self, sampler: SamplerBase, gen_file_path: str) -> EvalResult:
        prompt_messages_list = []
        for row in self.examples:
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE.format(**row), role="user")
            ]
            prompt_messages_list.append(prompt_messages)

        response_texts = sampler(prompt_messages_list, gen_file_path)

        results = []
        for i, response_text in enumerate(response_texts):
            row = self.examples[i]
            prompt_messages = prompt_messages_list[i]

            score, extracted_answer, answer_extracted = self._score_answer(
                response_text, row["Answer"]
            )

            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )

            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            result = SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={
                    "answer_extracted": answer_extracted,
                    "answer_normalized": bool(extracted_answer),
                },
            )
            results.append(result)

        return common.aggregate_results(results)
