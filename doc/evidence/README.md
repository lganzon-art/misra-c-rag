# MISRA C RAG capstone evidence

This folder records the prompt comparison, a retrieval failure and its fix, two cache timing examples, and guardrail behavior. The screenshots are examples from the runs shown; they do not establish average performance or prove that every possible attack is blocked. The prompt CSV files contain per-question results for 24 evaluation cases each.

## Architecture

[architecture.png](architecture.png) shows the deployed Google Cloud VM and Docker runtime, document indexing, guardrails, retrieval, response caching, answer generation, and operational tracing. The diagram is a design summary; the other files in this folder provide measured or observed evidence for specific components.

## Evaluation method

The prompt comparison uses the 24 questions and reference answers in `evaluation_dataset.json`. Cases may also include `reference_contexts`. The evaluation script, `evaluate.py`, asks the RAG assistant each question in a separate conversation, with response caching disabled for generation. It records the answer and retrieved excerpts, then scores each case with RAGAS. The three saved CSV files provide the per-case question, generated response, retrieved contexts, reference answer, and metric scores. The saved prompt text is in [`prompt/prompt.docx`](prompt/prompt.docx).

RAGAS computes five metrics: faithfulness compares answer claims with retrieved excerpts; answer relevancy judges how directly the response addresses the question; context precision judges the ranking of relevant retrieved excerpts; context recall judges whether retrieved excerpts cover the reference answer; and answer correctness compares the generated response with the reference answer. These are automated scores, not a manual certification of MISRA C correctness.

`evaluate.py` scores every case and metric separately, checks that each result is finite, and refuses to write a final CSV when any cell fails. Each saved prompt CSV contains **24 cases × 5 metrics = 120 finite values**. The results below are arithmetic means of the 24 case scores for each metric, rounded to six decimals. The CSVs do not record model, judge, corpus, code revision, or run configuration, so those settings cannot be independently verified from this evidence folder. The prompt comparison therefore shows observed runs, not an isolated causal test of prompt wording.

## Prompt iteration and results

**Files:** [`prompt/prompt.docx`](prompt/prompt.docx), [`prompt/ragas_results - prompt1.csv`](prompt/ragas_results%20-%20prompt1.csv), [`prompt/ragas_results - prompt2.csv`](prompt/ragas_results%20-%20prompt2.csv), [`prompt/ragas_results - prompt3.csv`](prompt/ragas_results%20-%20prompt3.csv), and the corresponding `prompt* score.png` screenshots.

Prompt 1 asks for answers grounded in retrieved excerpts. Prompt 2 adds a direct-answer-first format and examples. Prompt 3 adds instructions to preserve qualifiers such as “may” and “at least,” and includes a fourth example.

| Metric | Prompt 1 | Prompt 2 | Prompt 3 |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.866104 | 0.908978 | **0.952976** |
| Answer relevancy | 0.888030 | 0.893830 | **0.904753** |
| Context precision | 0.947917 | **0.948333** | 0.944676 |
| Context recall | **1.000000** | **1.000000** | **1.000000** |
| Answer correctness | 0.675861 | **0.796444** | 0.780985 |

| Comparison | Faithfulness | Answer relevancy | Context precision | Context recall | Answer correctness |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prompt 3 minus Prompt 1 | +0.086872 | +0.016723 | -0.003241 | 0.000000 | +0.105124 |
| Prompt 3 minus Prompt 2 | +0.043998 | +0.010923 | -0.003657 | 0.000000 | -0.015459 |

Prompt 3 has the highest faithfulness and answer relevancy in these runs. Prompt 2 has the highest context precision and answer correctness. Context recall ties across all three. Present Prompt 3 as a choice favoring faithfulness and relevancy, with the correctness tradeoff stated explicitly.

## Debugged retrieval failure

**Files:** [`bug_fix/bug.png`](bug_fix/bug.png) and [`bug_fix/bug_fix.png`](bug_fix/bug_fix.png).

**Question:** “What do advisory Rules 2.3 through 2.7 have in common?”

The earlier trace includes Rule 2.7 but does not show all requested Rules 2.3–2.6 among the visible retrieved excerpts. The later trace shows Rules 2.3, 2.4, 2.5, 2.6, and 2.7 in order. The visible failure is incomplete evidence for a range question. The retriever was changed to expand named rules and rule ranges to their corresponding sections, while other questions continue through hybrid BM25 and Qdrant retrieval with FlashRank reranking.

These screenshots prove improved **retrieval coverage for this question**. They do not by themselves show a before/after generated answer or a controlled aggregate metric change. If the presentation claims answer improvement, pair these traces with the corresponding answer outputs or evaluation rows.

## Index cache

**Files:** [`cache(embedding)/no_cache.png`](cache(embedding)/no_cache.png) and [`cache(embedding)/with_cache.png`](cache(embedding)/with_cache.png).

The LangSmith index summaries show the same 600 document chunks and embedding model in a local test. On the first preparation, `chunks_cache_hit` and `vector_index_cache_hit` are false; document loading and chunking take 1,613.77 ms, embedding and indexing take 21,107.34 ms, and total index preparation takes **23,961.6 ms**. On reuse, both hit flags are true; chunking and embedding/indexing are reported as 0 ms, while total index preparation still takes **2,548.06 ms**. The measured total reduction is **21.41 s, about 89.4%** for these traces. Use 23.96 s → 2.55 s for total index preparation; 0 ms applies only to the skipped substeps.


## Response cache

**Files:** [`cache(latency)/no cache.png`](cache(latency)/no%20cache.png) and [`cache(latency)/with_cache.png`](cache(latency)/with_cache.png).

The LangSmith request traces show **4.74 s** without a response-cache hit and **1.70 s** with a hit, a **3.04 s (about 64.1%)** reduction in those two requests. Retrieval still runs in both traces; the cache hit skips the chat model answer-generation call. These are single-request examples, not a benchmark distribution. The traces show different retrieval durations, so attribute the missing chat-model call to the cache, while treating the total latency difference as an observed comparison rather than a guaranteed saving.

## Guardrails

**Files:** [`guardrail/QnA.png`](guardrail/QnA.png) and [`guardrail/log.png`](guardrail/log.png).

The UI screenshot shows an email address masked before display, a permitted MISRA C answer with a citation, an unsafe request blocked by the safety guardrail, and an off-topic question blocked. The log screenshot contains an `allowed` event with `pii_masked: true`, a `blocked` event from `OpenAI moderation`, and two `blocked` events from `RestrictToTopic`. The log records event metadata rather than the raw questions or email address.

The “ignore all previous instructions” example in the screenshot is logged as **`RestrictToTopic`**, with PII masking. It is evidence that the request was blocked, but **not** evidence that `DetectJailbreak` fired. A separate log event naming `DetectJailbreak` would be needed to make that narrower claim.


## Limitations and next steps

- **Evaluation scope:** The prompt results cover 24 questions. The saved CSV files do not record model and code versions, so future comparisons should save a run manifest and reuse the same dataset, corpus, model settings, and judge configuration. Review reference answers and inspect low-scoring cases manually.
- **Retrieval proof:** The before and after screenshots show improved coverage for one rule-range question. Capture the corresponding answers and repeat the test on other ranges before claiming a broader correctness improvement.
- **Guardrail proof:** The recorded attack-like question was blocked by `RestrictToTopic`; the screenshots do not show `DetectJailbreak` firing. Add a separate trace or log event if a jailbreak-detector-specific claim is needed. Input screening does not mask uploaded documents.
- **Deployment:** The demonstration VM uses an HTTP URL and a local persistent disk. Add HTTPS and suitable access controls for ongoing public use, and back up the `data/` directory. A temporary external IP can change after VM lifecycle events.
- **Application state:** Conversation history is held in the Streamlit session, so a browser or server restart may lose it. Persist conversations separately if users need long-term history.
