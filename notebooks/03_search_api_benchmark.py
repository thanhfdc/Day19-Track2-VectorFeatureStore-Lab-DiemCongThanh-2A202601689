# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3 — FastAPI `/search` Endpoint + Latency Benchmark
#
# **Stack:** FastAPI + uvicorn + httpx (client). Searcher từ `app/search.py`.
# Maps to slide §7 (Production Patterns) + deliverable bullets 1, 4.
#
# > Mục tiêu: bọc `Searcher` thành REST API, đo P50/P95/P99 latency, đảm bảo
# > P99 < 50 ms cho hybrid mode (rubric threshold).

# %%
import _setup  # noqa: F401
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

# %% [markdown]
# ## 1. Khởi động API app trong notebook
#
# Trong production thực tế, bạn sẽ chạy `make api` ở terminal riêng. Notebook
# này dùng `TestClient` để gọi đúng FastAPI route `/search` mà không cần mở port.

# %%
ROOT = Path(_setup.__file__).resolve().parent.parent
from app.main import app

# TestClient enters FastAPI lifespan and builds Searcher once, while avoiding
# port conflicts in Jupyter.
api_client_cm = TestClient(app)
api_client = api_client_cm.__enter__()
print(api_client.get("/healthz").json())

# %% [markdown]
# ## 2. Single query — kiểm tra response shape

# %%
r = api_client.get("/search", params={"q": "cloud computing tự động mở rộng", "mode": "hybrid"})
r.raise_for_status()
body = r.json()
print(f"latency_ms: {body['latency_ms']:.1f}")
print(f"top-3 hits:")
for h in body["hits"][:3]:
    print(f"  {h['doc_id']:>14}  score={h['score']:.4f}  {h['title']}")

# %% [markdown]
# ## 3. TODO — Latency benchmark (100 queries × 3 modes)
#
# Dùng 50 golden queries × 2 reps = 100 calls/mode. Ghi nhận latency từ
# `body["latency_ms"]` (server-side, đã trừ network) HOẶC từ wall-clock httpx
# (bao gồm network) — note: rubric assert P99 < 50ms áp dụng cho server-side.
#
# Output: bảng P50/P95/P99 cho 3 mode.

# %%
import json

DATA = ROOT / "data"
golden = [json.loads(l) for l in (DATA / "golden_set.jsonl").open(encoding="utf-8")]


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def benchmark_mode(mode: str, reps: int = 10) -> dict[str, float]:
    server_latencies: list[float] = []
    wall_latencies: list[float] = []
    # Warm up ONNX/embedding internals and route code before tail-latency measurement.
    for q in golden:
        api_client.get("/search", params={"q": q["query"], "mode": mode})
    for _ in range(reps):
        for q in golden:
            t0 = time.perf_counter()
            r = api_client.get("/search", params={"q": q["query"], "mode": mode})
            wall_latencies.append((time.perf_counter() - t0) * 1000)
            server_latencies.append(r.json()["latency_ms"])
    return {
        "p50_server": percentile(server_latencies, 0.50),
        "p95_server": percentile(server_latencies, 0.95),
        "p99_server": percentile(server_latencies, 0.99),
        "p99_wall":   percentile(wall_latencies, 0.99),
    }


print(f"  {'mode':10}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'P99(wall)':>9}")
results = {}
for mode in ("keyword", "semantic", "hybrid"):
    res = benchmark_mode(mode)
    results[mode] = res
    print(f"  {mode:10}  {res['p50_server']:>5.1f}ms  {res['p95_server']:>5.1f}ms  "
          f"{res['p99_server']:>5.1f}ms  {res['p99_wall']:>7.1f}ms")

# %% [markdown]
# ## 4. Rubric assertion — hybrid P99 server-side < 50ms

# %%
hybrid_p99 = results["hybrid"]["p99_server"]
print(f"Hybrid P99 server-side: {hybrid_p99:.1f}ms")
if hybrid_p99 < 50:
    print(f"PASS — hybrid P99 < 50ms ({hybrid_p99:.1f}ms)")
else:
    print(f"WARN — hybrid P99 >= 50ms ({hybrid_p99:.1f}ms)")
    print("  Possible causes: cold cache, fastembed model not warm yet, or RRF depth=50 is too aggressive")
    print("  Check: re-run benchmark after 10 warm-up queries; or reduce RRF depth")

# %% [markdown]
# ## 5. Cleanup — close the API client

# %%
api_client_cm.__exit__(None, None, None)
print("API client closed")

# %% [markdown]
# ## Deliverable evidence
#
# 1. Output cell 2: 1 single hybrid query response with `top-3 hits`.
# 2. Output cell 3: latency table P50/P95/P99 for keyword/semantic/hybrid.
# 3. Output cell 4: hybrid P99 < 50ms PASS.
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** the FastAPI scaffolding (route definition, Pydantic
# response model, lifespan handler). AI generates this perfectly given the
# spec "GET /search?q=str&mode=Literal[...] returning SearchResponse with
# latency_ms field". `app/main.py` is exactly that pattern — review the diff,
# don't write it from scratch.
#
# **Think hard yourself:** *what to measure*. Server-side latency vs wall-clock
# vs client-side. P50 vs P95 vs P99. Cold vs warm. Single user vs concurrent.
# These are *judgement* decisions: nếu rubric chỉ check P99, optimization sẽ
# hướng vào tail latency, không phải mean. Đừng nhờ AI quyết định metric —
# chỉ nhờ implement metric đã chọn.
