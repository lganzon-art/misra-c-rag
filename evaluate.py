from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import tracing_context
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.run_config import RunConfig
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from rag import MisraRag, corpus_fingerprint


METRIC_FACTORIES = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
    "answer_correctness": AnswerCorrectness,
}


def load_cases(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("items", raw.get("questions", []))
    cases = []
    for i, row in enumerate(raw):
        question = row.get("question") or row.get("user_input")
        reference = row.get("reference") or row.get("reference_answer")
        if not question or not reference:
            raise ValueError(f"Case {i} needs question and reference/reference_answer fields")
        case = {"question": question, "reference": reference}
        reference_contexts = row.get("reference_contexts")
        if reference_contexts is not None:
            if not isinstance(reference_contexts, list) or not all(
                isinstance(context, str) for context in reference_contexts
            ):
                raise ValueError(f"Case {i} reference_contexts must be a list of strings")
            case["reference_contexts"] = reference_contexts
        cases.append(case)
    if not cases:
        raise ValueError("Evaluation dataset contains no cases")
    return cases


def load_saved_samples(path: Path, cases: list[dict]) -> list[dict]:
    """Reuse generated answers while ensuring they belong to this exact dataset."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != len(cases):
        raise ValueError(f"Saved samples must be a list of {len(cases)} cases")
    samples = []
    for index, (row, case) in enumerate(zip(raw, cases)):
        if not isinstance(row, dict):
            raise ValueError(f"Saved sample {index + 1} must be an object")
        if row.get("user_input") != case["question"] or row.get("reference") != case["reference"]:
            raise ValueError(f"Saved sample {index + 1} does not match the dataset question/reference")
        if (row.get("reference_contexts") or []) != (case.get("reference_contexts") or []):
            raise ValueError(f"Saved sample {index + 1} does not match the dataset reference contexts")
        if not isinstance(row.get("response"), str):
            raise ValueError(f"Saved sample {index + 1} needs a string response")
        contexts = row.get("retrieved_contexts")
        if not isinstance(contexts, list) or not all(isinstance(context, str) for context in contexts):
            raise ValueError(f"Saved sample {index + 1} needs a list of retrieved context strings")
        sample = {
            "user_input": case["question"],
            "response": row["response"],
            "retrieved_contexts": contexts,
            "reference": case["reference"],
        }
        if case.get("reference_contexts"):
            sample["reference_contexts"] = case["reference_contexts"]
        samples.append(sample)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the MISRA C RAG assistant with RAGAS")
    parser.add_argument("dataset", type=Path, help="JSON list or items object; question, reference, optional reference_contexts")
    parser.add_argument("--samples", type=Path, default=None,
                        help="Score saved .samples.json responses without running the RAG pipeline")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV path; defaults to ragas_results.csv for the full dataset or ragas_results-5Q.csv for the five-question subset")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per metric call in seconds")
    parser.add_argument("--retries", type=int, default=3, help="RAGAS retries per metric call")
    parser.add_argument("--attempts", type=int, default=2, help="Outer attempts for a failed case/metric")
    args = parser.parse_args()
    if args.samples is not None and args.output is None:
        parser.error("--samples requires --output so the existing results are not overwritten")
    if args.output is None:
        args.output = Path("ragas_results-5Q.csv" if args.dataset.stem == "evaluation_dataset-5Q" else "ragas_results.csv")

    if args.timeout < 1 or args.retries < 0 or args.attempts < 1:
        parser.error("--timeout must be positive; --retries must be non-negative; --attempts must be positive")

    cases = load_cases(args.dataset)
    if args.samples is not None:
        samples = load_saved_samples(args.samples, cases)
        print(f"Scoring {len(samples)} saved responses from {args.samples}", flush=True)
    else:
        rag = MisraRag(corpus_hash=corpus_fingerprint(), response_cache_enabled=False)
        generation_started = time.monotonic()
        print(f"Generating answers and contexts: 0/{len(cases)} (0.0%) elapsed=0s", flush=True)
        samples = []
        for index, case in enumerate(cases):
            result = rag.ask(
                case["question"],
                thread_id=f"ragas-eval-{index}",
                trace_source="evaluation-generation",
                trace_metadata={"case_index": index + 1},
            )
            sample = {
                "user_input": case["question"],
                "response": result["answer"],
                "retrieved_contexts": [item["text"] for item in result["contexts"]],
                "reference": case["reference"],
            }
            if case.get("reference_contexts"):
                sample["reference_contexts"] = case["reference_contexts"]
            samples.append(sample)
            elapsed = time.monotonic() - generation_started
            done = index + 1
            print(
                f"Generating answers and contexts: {done}/{len(cases)} "
                f"({done / len(cases):.1%}) elapsed={elapsed:.0f}s",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples_path = args.output.with_suffix(".samples.json")
    if args.samples is None or samples_path.resolve() != args.samples.resolve():
        samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Score each cell separately so failed judge calls cannot disappear from the mean.
    selected_eval_model = os.getenv("EVAL_MODEL", "gpt-6-luna")
    eval_options = {"model": selected_eval_model, "temperature": 0}
    if selected_eval_model.startswith("gpt-6-luna"):
        # RAGAS changes temperature per metric (usually 0.01 or 0.3).
        # Luna accepts those values only when reasoning is disabled.
        eval_options["reasoning_effort"] = "none"

    llm = ChatOpenAI(**eval_options)
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    run_config = RunConfig(
        timeout=args.timeout,
        max_retries=args.retries,
        max_wait=30,
        max_workers=1,
        log_tenacity=True,
    )
    scored = [
        {key: value for key, value in sample.items() if key != "reference_contexts" or value}
        | {metric: None for metric in METRIC_FACTORIES}
        for sample in samples
    ]
    errors = []
    checkpoint_path = args.output.with_suffix(".checkpoint.json")
    total_cells = len(samples) * len(METRIC_FACTORIES)
    completed_cells = 0
    started_at = time.monotonic()
    started_at_iso = datetime.now(timezone.utc).isoformat()
    print(
        f"Scoring {len(samples)} cases × {len(METRIC_FACTORIES)} metrics ({total_cells} cells) "
        f"started={started_at_iso}", flush=True
    )

    for case_index, sample in enumerate(samples):
        for metric_name, metric_factory in METRIC_FACTORIES.items():
            if scored[case_index][metric_name] is not None:
                continue
            error_messages = []
            for attempt in range(1, args.attempts + 1):
                try:
                    with tracing_context(
                        tags=["evaluation-judge", metric_name],
                        metadata={"case_index": case_index + 1, "metric": metric_name},
                    ):
                        result = evaluate(
                            dataset=EvaluationDataset.from_list([sample]),
                            metrics=[metric_factory()],
                            llm=llm,
                            embeddings=embeddings,
                            run_config=run_config,
                            raise_exceptions=True,
                            show_progress=False,
                        ).to_pandas()
                    value = float(result.iloc[0][metric_name])
                    if not math.isfinite(value):
                        raise ValueError(f"RAGAS returned non-finite {metric_name}: {value!r}")
                    scored[case_index][metric_name] = value
                    break
                except Exception as exc:
                    error_messages.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if scored[case_index][metric_name] is None:
                errors.append({
                    "case_index": case_index,
                    "metric": metric_name,
                    "question": sample["user_input"],
                    "errors": error_messages,
                })
            completed_cells += 1
            status = "ok" if scored[case_index][metric_name] is not None else "failed"
            elapsed_seconds = time.monotonic() - started_at
            print(
                f"[{completed_cells}/{total_cells} | {completed_cells / total_cells:.1%} | "
                f"elapsed {elapsed_seconds:.0f}s] "
                f"case {case_index + 1} {metric_name}: {status}",
                flush=True,
            )
            checkpoint_path.write_text(
                json.dumps({
                    "samples": scored,
                    "errors": errors,
                    "status": {
                        "state": "running",
                        "started_at": started_at_iso,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "completed_cells": completed_cells,
                        "total_cells": total_cells,
                        "completion_percent": round(completed_cells / total_cells * 100, 1),
                        "elapsed_seconds": round(elapsed_seconds, 1),
                    },
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    if errors:
        error_path = args.output.with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(
            f"Evaluation incomplete: {len(errors)} metric cells failed. "
            f"See {error_path.resolve()} and {checkpoint_path.resolve()}; no final CSV was written."
        )

    frame = pd.DataFrame(scored)
    missing = [
        (index, metric)
        for index, row in frame.iterrows()
        for metric in METRIC_FACTORIES
        if not math.isfinite(float(row[metric]))
    ]
    if missing:
        raise RuntimeError(f"Internal completeness check failed for cells: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame[list(METRIC_FACTORIES)].mean(numeric_only=True).to_string())
    print(f"\nScored {len(frame)} cases × {len(METRIC_FACTORIES)} metrics = {len(frame) * len(METRIC_FACTORIES)} finite values")
    print(f"\nWrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
