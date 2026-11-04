# CLAUDE.md

## Project
LLM inference server with continuous batching. I write the scheduler, the KV-cache memory manager, streaming API and benchmarks. Two backends: llama.cpp (baseline, Phases 0-6) and my own PyTorch model with a paged (block-based) KV cache and a C++ paged-attention function (Phase 7). Full roadmap: PLAN.md.

## Stack
- Python 3.11+, asyncio
- FastAPI + uvicorn (HTTP, SSE streaming)
- llama-cpp-python for the baseline backend (pin the exact version in requirements.txt)
- pytest for tests
- Phase 7 only: PyTorch for my own model forward pass; a C++ extension for paged attention. The binding/build tool is chosen in Phase 7.5 after reading the installed PyTorch version's documentation.
- No other dependencies without asking me first.

## Layout
- src/server/api.py             HTTP endpoints only, no scheduling logic
- src/server/scheduler.py       request queue + continuous batching loop
- src/server/kv_budget.py       KV-cache admission/accounting
- src/server/metrics.py         per-request timing and counters
- src/server/backend/base.py    Backend interface
- src/server/backend/fake.py    deterministic fake backend for tests
- src/server/backend/llamacpp.py  baseline real backend
- src/server/torch_model.py     my own Llama-style forward pass (Phase 7)
- src/server/block_allocator.py fixed-size KV block allocator and block tables (Phase 7)
- src/server/backend/paged.py   Backend adapter using the paged KV cache (Phase 7)
- csrc/paged_attention.cpp      C++ paged attention, decode, CPU (Phase 7.5)
- bench/loadgen.py, bench/report.py
- tests/ (including tests/test_paged_attention.py in Phase 7)
- The build file for the C++ extension is decided in Phase 7.5; ask me before creating it.

## Model
- GGUF file: models/qwen2.5-0.5b-instruct-q4_k_m.gguf (Qwen2.5 0.5B Instruct, Q4_K_M; not in git)

## Commands
- Install: pip install -r requirements.txt
- Test: pytest -q
- Run server: uvicorn server.api:app --app-dir src
(Update this section as commands are actually created. Never invent commands.)

## Rules for code (important)
1. Write only the code the current phase in PLAN.md needs. No extra features, options, flags, or "future-proofing".
2. No unused functions, imports, parameters, or config fields.
3. No comments that restate the code. Comment only the non-obvious "why".
4. No docstrings longer than one line.
5. Do not add try/except unless the failure case is named and tested.
6. Do not add new files not listed in Layout without asking.
7. Prefer small functions. Keep the scheduler readable in one screen where possible.

## Rules against hallucination
1. Never guess a library API. Before using llama-cpp-python, FastAPI, PyTorch or any C++ binding function, check the installed version: read its source or run `python -c "help(...)"` / `pip show`. Names in these libraries have changed between versions.
2. If unsure whether something exists, say so and check. Do not write code "that probably works".
3. Never report a benchmark number you did not measure. All numbers come from bench output files.
4. If a phase's acceptance criteria are not met, say so. Do not mark it done.
5. For Phase 7: the plain Python version of paged attention is the source of truth. The C++ output must match it. My forward pass must match a reference implementation's logits. Never claim a speedup or memory saving that was not measured.

## Workflow
- Work one phase at a time from PLAN.md. Stop at the end of each phase and show: what changed, test output, what is next.
- Write the test first when the behavior is clear (scheduler logic is tested against the fake backend, not a real model).
- Run `pytest -q` before saying a phase is finished.
- Ask me before: adding dependencies, changing the Backend interface, changing the benchmark methodology.

## Benchmark rules
- Baseline (naive, one request at a time) and continuous batching use the same backend, model, prompts, max_tokens, and sampling settings (greedy / fixed seed).
- Load is open-loop (requests arrive on a fixed schedule regardless of completions).
- Record hardware, model file, quantization, and library versions (llama-cpp-python, PyTorch) with every result.
- Comparisons between the llama.cpp backend and my own backend must state that the two backends differ in implementation; do not claim my backend is faster than llama.cpp unless measured.
