# Benchmarks — Qwen3.8-27B on RTX 3080 Ti + RTX 3060 (24GB)

All numbers measured on the author's machine: RTX 3080 Ti (12GB, PCIe x16) + RTX 3060 (12GB, PCIe x4), llama.cpp build `b31b71f3a` (2026-09-09, CUDA 13), GGUF **IQ4_XS ~14.6–15.5GB**, flash attention on, `-b 256 -ub 64`.

The model: **Qwen3.8-27B**, hybrid qwen35 architecture — 65 blocks (full attention on every 4th layer + 48 Mamba/SSM layers), MTP (NextN) draft embedded in some GGUFs, native context **262,144**. Because only ~17 layers carry attention KV, the cache is tiny: **~36 KB/token at q8_0, ~18 KB/token at q4_0** — that's why 262k fits at all.

Realistic prompt test: **94,164-token** real book text with a needle placed at 62% depth, 256 max output tokens, via the OpenAI API.

## Preset comparison

| Preset | Context | KV | Split | VRAM used | Prompt processing @ 94k | Generation @ 94k depth | Generation (shallow chat) |
|---|---|---|---|---|---|---|---|
| `fast` (MTP) | 98,304 | q8_0/q8_0 | tensor | ~22 GB | — | — | **~52 tok/s** (draft acceptance 0.62) |
| `ctx` | 163,840 | q8_0/q8_0 | tensor | ~21 GB | 459 tok/s | 19.9 tok/s | 26–29 tok/s |
| `max` | **262,144** | q4_0/q4_0 | **layer** | **22.5 GB** | **481 tok/s** | 15.0 tok/s | 25.8 tok/s |

**Takeaways**
- Going 160k → 262k costs almost nothing in prompt processing (it's even faster — quantized KV reads less bandwidth) and ~25% generation speed *at deep context only*. Shallow chat is unchanged.
- **Needle test at 94k with q4_0 KV: found** (`ZX-QUALITY-CODE-7743` retrieved exactly) on both the q8 and q4 presets.
- MTP (52 tok/s) does not fit next to a 262k KV cache — the 1.9GB draft tips it over 24GB. Only the 98k presets get MTP.
- `q5_1` KV @ 262k misses the fit by less than the available headroom — the largest safe step above q4_0 at 262k does not exist on 24GB.
- q8_0 KV hard ceiling: **163,840**. 196,608 fails during split-buffer allocation (null buffer = OOM).

## llama.cpp build `b31b71f3a` bugs (all reproduced live)

1. **Quantized KV + tensor split → crash on first decode** once the KV buffer is large:
   `ggml-backend-meta.cpp:537: GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN) failed`
   (also `ggml-backend-meta.cpp:1704: GGML_ASSERT(bufs.back() != nullptr)`).
   Works at `-c 8192`, crashes at 94k+/262k with q4_0/q4_0 and q8_0/q4_0. Workaround: **`-sm layer`**.
2. **Auto-fit is not implemented for tensor split**: `llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort` — the fitter also seems to ignore quantized KV when estimating. Workaround: **`-fit off`** when you know the config fits.
3. **Mixed KV types (q8_0 K + q4_0 V)** — the standard community recipe for long context — cannot work on this build: tensor split hits bug 1; layer split falls back to CPU for the quantized-V op (≈690% CPU, GPU idle, unusable).
4. Uniform **q4_0/q4_0 with layer split works** and is what the 262k preset ships.

On stock builds your mileage will differ — the mixed-KV recipe reportedly works upstream, and it would give a higher-quality 262k than uniform q4_0.

## How the presets were chosen

- 98k + MTP: the speed recipe (~52 tok/s real chat prompts, 256-token generations, draft acceptance 0.62). Everything on q8 KV.
- 160k: largest context that fits with q8_0 KV (196k verified to fail).
- 262k: model's native maximum. Requires the q4_0 KV + layer split + `-fit off` workarounds above; verified by a 94k needle test and full prompt-processing run.
