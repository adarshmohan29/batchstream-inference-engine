# PLAN.md

Goal: an inference server whose scheduler and KV memory manager I wrote, shown to beat naive one-at-a-time serving on throughput and tail latency, plus my own backend with block-based (paged) KV memory and a C++ paged-attention function.

## Key design decision (read first)
llama.cpp manages its own KV cache internally. With it as the backend (Phases 0-6), my "memory manager" is an admission controller: it tracks token budget (prompt + max_tokens per request against total context) and refuses or queues requests that do not fit. It is NOT block/page-level KV management.
Phase 7 builds my own model forward pass in PyTorch with a block-based KV allocator, so the memory manager becomes real block-level management. The C++ part is one bounded function: paged attention for decode on CPU.

## Concepts to understand before coding
- Prefill: process the whole prompt (compute-heavy). Decode: generate one token per step per sequence (memory-bound).
- Continuous batching: after every decode step, finished sequences leave and waiting ones join. No waiting for the whole batch to finish.
- Metrics: TTFT (time to first token), inter-token latency, end-to-end latency, tokens/s.
- Backpressure: when the queue is full, reject fast (HTTP 429) instead of growing unbounded.
- Paged KV cache: KV memory is split into fixed-size blocks; each sequence has a block table mapping its token positions to blocks, so memory is not reserved for the full context up front.

---

## Phase 0: Setup
- Create repo, venv, folder layout from CLAUDE.md, requirements.txt, pytest running with one trivial test.
- Choose a small GGUF model (e.g. ~0.5B-1B parameters, 4-bit quantized). Record exact file name.
Done when: `pytest -q` passes; model file path is in README or CLAUDE.md.

## Phase 1: Backend interface + fake backend
- Define Backend in base.py. It must support: starting a sequence from prompt tokens, running one step over all active sequences and returning one new token each, and releasing a sequence.
- Implement fake.py: returns deterministic tokens, ends a sequence after N tokens, sleeps a fixed time per step.
Done when: unit tests show the fake backend steps multiple sequences and ends them correctly.

## Phase 2: Naive baseline
- scheduler.py: FIFO, one request fully finished before the next starts (uses the same Backend).
- metrics.py: record arrival, first-token, finish times per request.
Done when: tests with fake backend show requests run strictly in order; metrics computed correctly.

## Phase 3: Continuous batching scheduler (core of project)
- Loop: admit waiting requests while capacity allows -> run one backend step -> emit tokens -> remove finished -> repeat.
- Config: max concurrent sequences.
Done when, with fake backend:
  - a short request arriving behind a long one finishes before the long one (proves no head-of-line blocking);
  - total time for N concurrent requests is less than naive.

## Phase 4: KV budget + backpressure
- kv_budget.py: total token capacity; reserve prompt_len + max_tokens on admission; release on finish.
- Bounded waiting queue; queue full -> reject.
- Request that can never fit -> reject immediately.
Done when: tests cover admit, queue-when-full, reject-when-queue-full, reject-too-large, and release after finish.

## Phase 5: HTTP API + streaming
- Endpoint to submit a prompt with max_tokens; stream tokens via SSE; 429 on rejection.
- Handle client disconnect by cancelling the request and releasing its KV budget.
Done when: manual curl streams tokens; a disconnect test confirms budget is released.

## Phase 6: Real backend (llama.cpp)
- Implement llamacpp.py against the pinned llama-cpp-python version. Verify every API call against installed source/help before using it.
- Batch multiple sequences in one decode call using separate sequence IDs.
- First check: output of a single request matches the naive path with greedy sampling.
Done when: same prompts produce identical greedy output in naive and batched modes, and batched mode is faster under concurrent load.

## Phase 7: Own backend with paged KV cache and C++ paged attention
Do these sub-phases in order. Stop after each one. Ask me before changing the Backend interface.

### 7.0 Model and dependencies
- I choose and download a small Llama-style model in a format loadable by PyTorch myself. Record filename, architecture and exact dependency versions in CLAUDE.md. Do not download anything.
Done when: model and dependency versions are recorded and installed.

### 7.1 Own forward pass, contiguous KV, one sequence (torch_model.py)
- Implement the model's forward pass myself and a simple contiguous KV cache.
- Compare against a reference implementation on the same prompt.
Done when: logits match the reference within a stated tolerance and greedy output tokens are identical.

### 7.2 Block allocator (block_allocator.py)
- Fixed block size, free list, block table per sequence, allocate and free.
Done when: tests cover allocate, free, out-of-blocks, and no leaked blocks after many allocate/free cycles.

### 7.3 Paged attention in plain Python
- Attention for one new token that reads K/V through the block table.
Done when: output matches 7.1 exactly (within the same tolerance) for several sequence lengths, including lengths that do not fill the last block.

### 7.4 Backend adapter (paged.py)
- Implement the Backend interface using the paged KV cache; support several sequences stepped together.
- Replace the token-based KV budget with a block-based one for this backend.
Done when: batched greedy output equals one-at-a-time greedy output, and the scheduler runs with this backend end to end.

### 7.5 C++ paged attention
- Implement the same function as 7.3 in csrc/paged_attention.cpp with Python bindings. Read the installed PyTorch version's extension documentation before choosing the build approach; ask me before creating the build file.
Done when: C++ output matches the Python version within a stated tolerance across test cases.

### 7.6 Phase 7 measurements
- Memory used by paged blocks vs reserving a full context per request, for the same workload.
- Python vs C++ attention speed.
Done when: results are saved as files with hardware and version info. Report results even if C++ is not faster.

## Phase 8: Benchmarks
- loadgen.py: open-loop arrivals at several rates (e.g. 4-5 levels from light to overload), fixed random seed, fixed prompt set with varied lengths.
- For each rate and each mode (naive vs batched) and each backend (llama.cpp, own paged backend): tokens/s, TTFT p50/p99, end-to-end p50/p99, rejection count.
- report.py: write CSV and plots from the CSV only.
Done when: results folder has raw CSV, plots, and hardware/version info.

## Phase 9: Write-up
- README: architecture diagram, how to run, results table and plots, honest limitations (see design decision above).
- Include one failure analysis: what happens at overload and why.
- State clearly which results compare scheduling modes on one backend and which compare two different backends.

---

## Order of work
0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7.0 -> 7.1 -> 7.2 -> 7.3 -> 7.4 -> 7.5 -> 7.6 -> 8 -> 9.
Phases 1-5 need no model and no GPU, so they can be built and tested fast.
If time is short, Phase 7.5 and 7.6 can be skipped; 7.0-7.4 alone already give a real paged-KV backend.
