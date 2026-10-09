# qwen — Qwen3.8-27B engine for two 12GB GPUs (2× RTX 3060 class)

Run **4 different Qwen3.8-27B fine-tunes** (base/uncensored, pi, ThinkingCap, Swift) through a single llama.cpp server with **one command each** — including model switching, three context presets (98k / 160k / **262k**), and per-variant **thinking levels** (off/low/medium/xhigh).

```
qwen pi max xhigh        # pi tune, 262k context, maximum thinking
qwen cap                 # ThinkingCap, 98k, low thinking (extraction)
qwen code max low        # Swift tune, 262k, light thinking
```

**The point: real agentic work in a home lab on two 12GB cards.** Two RTX 3060 12GB run the pi coding agent on the `pi` tune all day, opencode/goose on the `code` tune, and full **262k-token document work on ~22.5GB of VRAM** — at the speeds measured below. This exact setup ran as a daily production machine, not a demo.

Tested hardware: **RTX 3080 Ti (12GB) + RTX 3060 (12GB)**, llama.cpp build `b31b71f3a` (2026-09-09, CUDA), GGUF **IQ4_XS ~15GB** per model. A dual-RTX-3060-12GB setup reported nearly identical speed on the same recipe (50.6 vs 52.1 tok/s, r/LocalLLaMA) — expect essentially the same results on two 3060s. Everything is measured — see [BENCHMARKS.md](BENCHMARKS.md).

---

## The idea: four tunes, four jobs

All four are fine-tunes of the **same Qwen3.8-27B** — they do the same tasks as the original model, but with **2–4× fewer thinking tokens for the same answers** (measured by their trainers: ThinkingCap −37%, Swift −58% on GPQA/LiveCodeBench; pi finishes a task with ≈4,650 thinking tokens vs ≈8,300 for the base). Each tune exists for a different job:

| Variant | Exact GGUF file (on disk, ~15GB each) | Upstream fine-tune | Built for | Thinking default |
|---|---|---|---|---|
| `fast` / `ctx` / `max` | `RVN-IQ4_XS-mtp.gguf` | [0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF) | **The Heretic-uncensored base** — general-purpose work where refusals hurt; the accuracy reference for long autonomous runs | off |
| `pi` | `bytkim_Qwen3.8-27B-pi-IQ4_XS.gguf` | [bytkim/Qwen3.8-27B-pi](https://huggingface.co/bytkim/Qwen3.8-27B-pi) | **Built specifically for the pi coding agent** — SFT+GRPO on successful pi sessions: repo reading, file edits, tool loops. ≈4,650 vs ≈8,300 thinking tokens per task; at `medium` it still matches the base model's `xhigh` quality | medium |
| `cap` | `ThinkingCap-Qwen3.8-27B-IQ4_XS.gguf` | [bottlecapai/ThinkingCap-Qwen3.8-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B) | **General work that needs brief thinking** — research, extraction, classification, text-to-SQL; best long-context recall (+2.3pp). −37% reasoning tokens, same answers | **low** |
| `code` | `Swift-Qwen3.8-27B-IQ4_XS.gguf` | [ukisai/Swift-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-Qwen3.8-27b) | **Built for the other coding agents** (opencode, goose) — −58% thinking tokens, ~2× faster, LiveCodeBench 77→82 | medium |

**Don't like the pi agent?** Point opencode or goose at the same server alias and run `qwen code` (Swift) — the one-alias design below means zero config changes. The tunes exist to make the SAME model faster at specific jobs; pick per job.

⚠️ **Don't run the thinking tunes with thinking fully off.** ThinkingCap was trained *with* thinking — disabling it entirely costs **−34pp** accuracy (the single biggest quality drop of any setting). The defaults in this script follow each tune's training: extraction tunes get `low`, agent tunes get `medium`, the base gets `off` for raw speed.

## Command reference

```
qwen                                  # usage
qwen <variant> [max] [off|low|medium|xhigh]

  fast            # base @ 98k, MTP, ~52 tok/s        (thinking: off)
  ctx             # base @ 160k, q8 KV                (thinking: off)
  pi              # pi tune @ 98k, MTP + pi agent     (thinking: medium)
  cap             # ThinkingCap @ 98k                 (thinking: low)
  code            # Swift @ 98k                       (thinking: medium)
  max             # base @ 262k, q4 KV                (thinking: off)

  pi max          # pi tune @ 262k                    (thinking: medium)
  code max low    # Swift @ 262k, light thinking
  cap off         # ThinkingCap, no thinking — max-speed batches
  pi xhigh max    # flag order doesn't matter
  high            # alias for xhigh (Qwen3.8 levels: low/medium/xhigh)

qwen status          # what's running: variant, thinking level, VRAM, last t/s
qwen stop            # kill the server (frees both GPUs)
qwen log [variant]   # tail a server log
```

Rules:
- **Running server + a new command = switch** (graceful stop → load, ~30–40s). Same model + new thinking level = quick reload. Same everything = no-op.
- **All variants answer under one alias** (`qwen3.8-27b`), so pi / opencode / goose / any OpenAI-compatible client keeps working no matter which tune is loaded.
- Zombies (dead servers whose parent was an `exec`-ing agent) are detected and ignored — they don't fool start/stop.

## Context presets — what actually fits in 24GB

| Preset | Context | KV cache | Split | VRAM | Speed (measured) |
|---|---|---|---|---|---|
| `fast` (MTP) | 98,304 | q8_0 / q8_0 | tensor | ~22 GB | **~52 tok/s** generation (draft acceptance 0.62) |
| `ctx` | 163,840 | q8_0 / q8_0 | tensor | ~21 GB | 26–29 tok/s · prompt 459 tok/s |
| `max` | **262,144** | **q4_0 / q4_0** | **layer** | 22.5 GB | 25.8 tok/s chat · prompt **481 tok/s** · 15.0 tok/s at 94k depth |

The 262k preset is the interesting one: the model is a **hybrid** (17 full-attention + 48 Mamba/SSM layers), so KV is only ~18 KB/token at q4_0 → 262k fits with ~1.8GB to spare. A 94,164-token needle-in-haystack test (needle at 62% depth) passes. Two build-specific workarounds are mandatory there — see the bugs section.

### The exact launch flags behind each preset (copy-paste)

All: `GGML_CUDA_DISABLE_GRAPHS=1`, `-ngl 99 -fa on --jinja -rea off --port 1234 --metrics`.

**fast (MTP, 98k → ~52 tok/s):**
```
-c 98304 -b 256 -ub 64 -sm tensor -ts 1/1 -ctk q8_0 -ctv q8_0 \
--spec-type draft-mtp --spec-draft-n-max 2 -ngld 99 -ctkd q4_0 -ctvd q4_0
```
(RVN and bytkim GGUFs have the MTP draft embedded — ThinkingCap/Swift don't.)

**ctx (160k — the q8 KV ceiling on this build):**
```
-c 163840 -sm tensor -ts 1/1 -ctk q8_0 -ctv q8_0
```

**max (262k → fits at 22.5GB):**
```
-c 262144 -b 256 -ub 64 -sm layer -ctk q4_0 -ctv q4_0 -fit off
```

Thinking level goes over `-rea off` as request-side/template data (see next section). The script wraps all of this — these are the raw flags if you'd rather roll your own.

## Thinking levels — how they work

The server starts with `-rea off` (template-level default) and thinking is enabled via:

```
--chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "low"}'
```

Because it's just request-side data, **any client can override it per request**:

```json
{"messages": [...], "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "xhigh"}}
```

Reasoning comes back cleanly separated in `message.reasoning_content`. Defaults per variant follow each tune's training (see table above). Community consensus for agent loops: **medium is the sweet spot** — "saving" tokens at low/off often costs extra agent rounds that erase the win (−28…34pp accuracy when fully off).

## llama.cpp bugs hit on build `b31b71f3a` (all verified live)

1. **Quantized KV + `-sm tensor` crashes on decode** with large KV: `GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)` / `bufs.back() != nullptr` in `ggml-backend-meta.cpp`. Small contexts pass. → the 262k preset uses `-sm layer`.
2. **Auto-fit aborts**: `llama_params_fit is not implemented for SPLIT_MODE_TENSOR` when the fitter decides the config doesn't fit → pass `-fit off` when you know it fits.
3. **q8 KV ceiling is 163,840** — 196k fails with a null-buffer assert (OOM in split allocation).
4. **q5_1 KV @ 262k does not fit** (misses by less than 1.8GB of headroom).
5. **Mixed KV types (q8 K + q4 V)** — the standard community recipe for 262k — is impossible on this build: tensor split crashes (bug 1), layer split falls back to CPU (~17 min per test). Should work on stock builds; upstream fixes would unlock a better-quality 262k.

⚠️ These findings are **build-specific**. External data point: [Captain-Tripps/qwen38-dual-rtx-3060](https://github.com/Captain-Tripps/qwen38-dual-rtx-3060) runs quantized KV + tensor split fine on a newer build (`b29c606e2`, 131k, q4_0 KV) — so bug 1 is a `b31b71f3a` regression, not llama.cpp-wide. Re-check before copying anything here onto other builds.

## Credits

- The `fast` speed recipe builds on the dual-RTX-3060 work logged by **[Captain-Tripps/qwen38-dual-rtx-3060](https://github.com/Captain-Tripps/qwen38-dual-rtx-3060)** (MIT) and the r/LocalLLaMA thread around it (50.6 tok/s on two 3060s; 52.1 here on 3080 Ti + 3060). Their repo is also the external confirmation that the split bugs above are build-specific.
- Everything else — the four-tune matrix, the 160k/262k presets, thinking levels, the switching engine, and the bug findings — is original to this repo and was **not** forked from anywhere.

## Adapting it

Everything lives in the `CONFIG` block at the top of the script: paths, port, alias, and the four GGUF paths. Requirements:

- llama.cpp built with CUDA + flash attention (tested on `b31b71f3a`; see bugs above for older/newer builds)
- 2 NVIDIA GPUs, ~24GB total VRAM (single GPU works with `-sm layer` and adjusted flags)
- ~60GB disk for the four IQ4_XS GGUFs (~15GB each, sources linked in the table above)
- The `pi` variant launches the [pi coding agent](https://github.com/badlogic/pi) after the server is up — delete that line if you don't use it

Licenses of the models (not this repo): pi = Apache-2.0 · Swift = free under $1M annual revenue · ThinkingCap = PolyForm Small Business + personal-use grant — **check before commercial use**.

## License

MIT — see [LICENSE](LICENSE).
