# CLAUDE.md

This repository's canonical agent usage guide is in `AGENTS.md`.

If you are an AI coding agent (including Claude-compatible tooling), read `AGENTS.md` first for:

- MCP purpose and architecture
- **Web UI** — LTX-WS Videofentanyl (`web_ui.py`, `web/`): library, multi-clip autocontinue/autoconcat, LoRA dropdown
- tool-level usage guidance
- **MLX-only model weights** (`dgrauet/ltx-2.3-mlx*` — never standard `Lightricks/LTX-2.3`)
- serving directors with **autocontinue** multi-segment workflows (~5s clips → longer deliverables)
- LTX-2.3 prompt optimization for chained sequences
- standard generation workflows and examples

Human-facing setup and CLI reference: [`README.md`](README.md).
