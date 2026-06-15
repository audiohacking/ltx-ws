# CLAUDE.md

This repository's canonical agent usage guide is in `AGENTS.md`.

If you are an AI coding agent (including Claude-compatible tooling), read **`AGENTS.md`** first for MCP, Web UI, and MLX weights.

When the user wants to **generate video**, **improve a prompt**, or **plan a longer clip**, also read **`DIRECTOR.md`** and act as the **Assistant Director**: interview when needed, transform prompts to gold, present a shot plan, then call the pipeline.

`AGENTS.md` covers:

- MCP purpose and architecture
- **Web UI** — LTX-WS Videofentanyl (`web_ui.py`, `web/`): library, multi-clip autocontinue/autoconcat, LoRA dropdown
- tool-level usage guidance
- **MLX-only model weights** (`dgrauet/ltx-2.3-mlx*` — never standard `Lightricks/LTX-2.3`)
- serving directors with **autocontinue** multi-segment workflows (~5s clips → longer deliverables)
- standard generation workflows and examples

`DIRECTOR.md` covers:

- Assistant Director workflow (intake → gold prompts → shot plan → generate → iterate)
- LTX-2.3 prompt craft and autocontinue segment writing
- when to ask the user vs proceed with stated assumptions

Human-facing setup and CLI reference: [`README.md`](README.md).
