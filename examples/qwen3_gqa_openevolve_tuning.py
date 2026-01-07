"""
Qwen3 GQA Attention Kernel Tuning with OpenEvolve
==================================================

This example tunes a Helion GQA causal attention kernel for Qwen3 shapes:
- Q: (B, H_q, S, D) with H_q=32
- K/V: (B, H_kv, S, D) with H_kv=8
- bf16 inputs, fp32 accumulation
"""

from __future__ import annotations

import math
import os
import sys
import multiprocessing
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import helion
import helion.language as hl

try:
    from helion.autotuner.openevolve_tuner import OpenEvolveTuner
except ImportError:
    print("Error: OpenEvolve not installed. Install with: pip install openevolve")
    sys.exit(1)

MOCK_MODE = "OPENAI_API_KEY" not in os.environ

DEVICE_IDS = list(range(torch.cuda.device_count())) or [0]
_EVAL_COUNT = 0


def _next_device() -> tuple[int, torch.device]:
    global _EVAL_COUNT
    device_id = DEVICE_IDS[_EVAL_COUNT % len(DEVICE_IDS)]
    _EVAL_COUNT += 1
    return device_id, torch.device(f"cuda:{device_id}")


def create_gqa_kernel_with_config(config: dict[str, Any]):
    block_sizes = [config["block_bh"], config["block_m"], config["block_n"]]

    @helion.kernel(
        static_shapes=True,
        config=helion.Config(
            block_sizes=block_sizes,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        ),
    )
    def gqa_attention(
        q_in: torch.Tensor,  # (B, H_q, S, D)
        k_in: torch.Tensor,  # (B, H_kv, S, D)
        v_in: torch.Tensor,  # (B, H_kv, S, D)
    ) -> torch.Tensor:
        B, H_q, S, D = q_in.shape
        Bk, H_kv, Sk, Dk = k_in.shape
        assert Bk == B
        assert Sk == S
        assert Dk == D
        group_size = H_q // H_kv
        assert group_size * H_kv == H_q

        head_dim = hl.specialize(D)
        q_view = q_in.reshape([B * H_q, S, head_dim])
        k_view = k_in.reshape([B * H_kv, S, head_dim]).transpose(1, 2)
        v_view = v_in.reshape([B * H_kv, S, head_dim])

        out = torch.empty_like(q_view)
        sm_scale = 1.0 / math.sqrt(head_dim)
        qk_scale = sm_scale * 1.44269504  # 1 / log(2), used with exp2

        for tile_bh, tile_m in hl.tile([q_view.size(0), S]):
            bh = tile_bh.begin
            b = bh // H_q
            hq = bh - b * H_q
            hkv = hq // group_size
            kv = b * H_kv + hkv

            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.full([tile_m], 1.0, dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            q = q_view[bh, tile_m, :]

            for tile_n in hl.tile(tile_m.end):
                k = k_view[kv, :, tile_n]
                qk = hl.dot(q, k, out_dtype=torch.float32)

                pred = tile_m.index[:, None] >= tile_n.index[None, :]
                qk = torch.where(pred, qk, float("-inf"))

                m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
                p = torch.exp2(qk)
                l_ij = torch.sum(p, -1)
                alpha = torch.exp2(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]

                v = v_view[kv, tile_n, :]
                p = p.to(v.dtype)
                acc = hl.dot(p, v, acc=acc)

                m_i = m_ij

            acc = acc / l_i[:, None]
            out[bh, tile_m, :] = acc.to(out.dtype)

        return out.view(q_in.size())

    return gqa_attention


def gqa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, H_q, S, D = q.shape
    _, H_kv, _, _ = k.shape

    group_size = H_q // H_kv
    k = k.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(B, H_q, S, D)
    v = v.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(B, H_q, S, D)

    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def evaluate_config(config: dict[str, Any]) -> float:
    try:
        kernel = create_gqa_kernel_with_config(config)

        device_id, device = _next_device()
        torch.cuda.set_device(device_id)

        B, H_q, H_kv, S, D = 2, 32, 8, 512, 128
        q = torch.randn(B, H_q, S, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, S, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, S, D, device=device, dtype=torch.bfloat16)

        y_helion = kernel(q, k, v)
        y_ref = gqa_reference(q, k, v)
        if not torch.allclose(y_helion, y_ref, rtol=1e-2, atol=1e-2):
            return 0.0

        from triton.testing import do_bench

        B, H_q, H_kv, S, D = 4, 32, 8, 4096, 128
        q = torch.randn(B, H_q, S, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, S, D, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, S, D, device=device, dtype=torch.bfloat16)

        time_ms = do_bench(lambda: kernel(q, k, v), warmup=10, rep=30)

        flops = 4 * B * H_q * S * S * D
        tflops = (flops / (time_ms * 1e-3)) / 1e12
        return tflops

    except torch.cuda.OutOfMemoryError:
        return 0.0
    except Exception as e:
        print(f"Config failed: {config}, Error: {e}")
        return 0.0


def mock_evaluate_config(config: dict[str, Any]) -> float:
    block_m = config["block_m"]
    block_n = config["block_n"]
    num_warps = config["num_warps"]

    score = 1.0
    score *= 1.0 - abs(block_m - 64) / 256
    score *= 1.0 - abs(block_n - 64) / 256
    score *= 1.0 - abs(num_warps - 4) / 8

    base = 150.0
    return max(0.0, base * score)


def main() -> None:
    print("=" * 70)
    print("Qwen3 GQA Attention Tuning with OpenEvolve")
    print("=" * 70)

    config_space = {
        "block_bh": [1, 2],
        "block_m": [16, 32, 64, 96, 128, 192, 256],
        "block_n": [16, 32, 64, 96, 128, 192, 256],
        "num_warps": [1, 2, 4, 8],
        "num_stages": [1, 2, 3, 4, 5],
    }

    if len(DEVICE_IDS) > 1:
        print(f"Using {len(DEVICE_IDS)} GPUs: {DEVICE_IDS}")
    else:
        print(f"Using 1 GPU: {DEVICE_IDS[0]}")

    if MOCK_MODE:
        print("\nRunning in MOCK MODE (no API key)")
        evaluate_fn = mock_evaluate_config
        max_evals = 20
    else:
        print("\nRunning in REAL MODE (GPU evaluation)")
        evaluate_fn = evaluate_config
        max_evals = 120

    tuner = OpenEvolveTuner(
        config_space=config_space,
        objective=evaluate_fn,
        max_evaluations=max_evals,
        population_size=20,
        temperature=0.7,
        verbose=True,
        initial_config={
            "block_bh": 1,
            "block_m": 64,
            "block_n": 64,
            "num_warps": 4,
            "num_stages": 2,
        },
        llm_models=[
            {
                "name": "gpt-5.2",
                "weight": 1.0,
                "api_base": "https://api.openai.com/v1",
                "api_key": "${OPENAI_API_KEY}",
            },
        ],
    )

    tuning_start = time.perf_counter()
    best_config = tuner.tune()
    tuning_time_s = time.perf_counter() - tuning_start
    print(f"Best config: {best_config}")
    print(f"Tuning time: {tuning_time_s:.2f}s")

    _write_report(best_config, tuning_time_s)


def _benchmark_kernel(
    kernel_fn,
    B: int,
    H_q: int,
    H_kv: int,
    S: int,
    D: int,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    device_id, device = _next_device()
    torch.cuda.set_device(device_id)
    q = torch.randn(B, H_q, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H_kv, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H_kv, S, D, device=device, dtype=dtype)

    from triton.testing import do_bench

    time_ms = do_bench(lambda: kernel_fn(q, k, v), warmup=25, rep=100)
    flops = 4 * B * H_q * S * S * D
    tflops = (flops / (time_ms * 1e-3)) / 1e12
    return {"avg_ms": float(time_ms), "tflops": float(tflops)}


def _make_sdpa_gqa():
    def sdpa_gqa(q, k, v):
        B, H_q, S, D = q.shape
        _, H_kv, _, _ = k.shape
        group_size = H_q // H_kv
        k = k.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(B, H_q, S, D)
        v = v.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(B, H_q, S, D)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    return sdpa_gqa


def _write_report(best_config: dict[str, Any], tuning_time_s: float) -> None:
    report_dir = Path("openevolve_artifacts") / "qwen3_gqa_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    B, H_q, H_kv, S, D = 4, 32, 8, 4096, 128
    baseline_kernel = create_gqa_kernel_with_config(
        {
            "block_bh": 1,
            "block_m": 64,
            "block_n": 64,
            "num_warps": 4,
            "num_stages": 2,
        }
    )
    tuned_kernel = create_gqa_kernel_with_config(best_config)
    sdpa_kernel = _make_sdpa_gqa()

    results = []

    eager = _benchmark_kernel(sdpa_kernel, B, H_q, H_kv, S, D)
    results.append(
        {
            "impl": "sdpa_eager",
            "avg_ms": eager["avg_ms"],
            "tflops": eager["tflops"],
            "tuning_time_s": None,
        }
    )

    try:
        compiled_sdpa = torch.compile(sdpa_kernel, fullgraph=True)
        compiled = _benchmark_kernel(compiled_sdpa, B, H_q, H_kv, S, D)
        results.append(
            {
                "impl": "sdpa_torch_compile",
                "avg_ms": compiled["avg_ms"],
                "tflops": compiled["tflops"],
                "tuning_time_s": None,
            }
        )
    except Exception as e:
        results.append(
            {
                "impl": "sdpa_torch_compile",
                "avg_ms": None,
                "tflops": None,
                "tuning_time_s": None,
                "note": f"compile_failed: {e}",
            }
        )

    helion_baseline = _benchmark_kernel(baseline_kernel, B, H_q, H_kv, S, D)
    results.append(
        {
            "impl": "helion_baseline",
            "avg_ms": helion_baseline["avg_ms"],
            "tflops": helion_baseline["tflops"],
            "tuning_time_s": None,
        }
    )

    helion_tuned = _benchmark_kernel(tuned_kernel, B, H_q, H_kv, S, D)
    results.append(
        {
            "impl": "helion_openevolve",
            "avg_ms": helion_tuned["avg_ms"],
            "tflops": helion_tuned["tflops"],
            "tuning_time_s": tuning_time_s,
        }
    )

    eager_ms = eager["avg_ms"]
    for row in results:
        if row.get("avg_ms") is not None:
            row["speedup_vs_eager"] = eager_ms / row["avg_ms"]
        else:
            row["speedup_vs_eager"] = None

    csv_path = report_dir / "qwen3_gqa_results.csv"
    md_path = report_dir / "qwen3_gqa_results.md"
    plot_path = report_dir / "qwen3_gqa_results.png"

    csv_lines = ["impl,avg_ms,tflops,speedup_vs_eager,tuning_time_s,note"]
    for row in results:
        csv_lines.append(
            ",".join(
                [
                    str(row.get("impl")),
                    str(row.get("avg_ms")),
                    str(row.get("tflops")),
                    str(row.get("speedup_vs_eager")),
                    str(row.get("tuning_time_s")),
                    str(row.get("note", "")),
                ]
            )
        )
    csv_path.write_text("\n".join(csv_lines))

    md_lines = [
        "# Qwen3 GQA Tuning Results",
        "",
        f"Shape: B={B}, H_q={H_q}, H_kv={H_kv}, S={S}, D={D}",
        "",
        "| impl | avg_ms | tflops | speedup_vs_eager | tuning_time_s | note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in results:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("impl")),
                    str(row.get("avg_ms")),
                    str(row.get("tflops")),
                    str(row.get("speedup_vs_eager")),
                    str(row.get("tuning_time_s")),
                    str(row.get("note", "")),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(md_lines))

    try:
        import matplotlib.pyplot as plt

        labels = [r["impl"] for r in results]
        ms_values = [r["avg_ms"] if r["avg_ms"] is not None else 0 for r in results]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(labels, ms_values)
        ax.set_ylabel("avg_ms (lower is better)")
        ax.set_title("Qwen3 GQA attention latency")
        ax.set_xticklabels(labels, rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
    except Exception as e:
        note_path = report_dir / "plot_error.txt"
        note_path.write_text(f"Plot generation failed: {e}")

    _update_blog_results(md_path, plot_path)


def _update_blog_results(md_path: Path, plot_path: Path) -> None:
    blog_path = Path("docs") / "openevolve_kernel_evolution_blog.md"
    if not blog_path.exists():
        return

    table_lines = []
    for line in md_path.read_text().splitlines():
        if line.strip().startswith("|"):
            table_lines.append(line)

    if not table_lines:
        return

    updated = time.strftime("%Y-%m-%d %H:%M:%S")
    image_block = ""
    if plot_path.exists():
        image_path = plot_path.resolve()
        image_rel = image_path.relative_to(Path.cwd())
        image_block = f"\n![Qwen3 GQA latency plot]({image_rel.as_posix()})"

    new_block = "\n".join(
        [
            "```",
            *table_lines,
            "```",
            f"Updated: {updated}",
            image_block,
        ]
    )

    text = blog_path.read_text()
    start_token = "<!-- RESULTS:START -->"
    end_token = "<!-- RESULTS:END -->"
    if start_token not in text or end_token not in text:
        return

    before, rest = text.split(start_token, 1)
    _, after = rest.split(end_token, 1)
    blog_path.write_text(
        before + start_token + "\n" + new_block + "\n" + end_token + after
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
