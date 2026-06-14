# PR: Embedded Web UI (LTX-WS Videofentanyl)

**Branch:** `webui` → `main`

## Summary

- Adds **LTX-WS Videofentanyl** — embedded React Web UI served from `server.py` (default `--web-ui`, same port as WebSocket).
- **Multi-clip autocontinue + autoconcat** in the browser (matches `videofentanyl --count N --autocontinue --autoconcat`); in-process generation on embedded server.
- **Clip library** with persistent index, delete, settings restore; fixes library state when starting new generations.
- **Live progress** — denoising step + ETA via SSE (`model_progress` from MLX tqdm hook).
- **LoRA dropdown** — default OmniNFT RL (`DEFAULT_LORA_URL` / `LTX_WS_DEFAULT_LORA`), auto-download via `/api/loras/ensure`, per-request `lora_specs` without `--enable-lora`.
- **ltx-2-mlx v0.14.9** backend compatibility; separate **t2v / i2v** pipeline instances for autocontinue conditioning.
- Optional standalone UI: `web_server.py` for remote WebSocket attachment.
- Docs: README, AGENTS.md, CLAUDE.md updated.

## Test plan

- [ ] `cd web && npm install && npm run build`
- [ ] `python server.py --model ltx-2.3-mlx-q8` — Web UI at `http://127.0.0.1:8765/`
- [ ] Single clip generate (no LoRA / with OmniNFT LoRA dropdown)
- [ ] ×2 clips, autocontinue + autoconcat — merged video in library, visual continuity
- [ ] Second generation — prior clips remain in library
- [ ] CLI parity: `python videofentanyl.py --server ws://127.0.0.1:8765/ws --prompt "…" --count 2 --autocontinue --autoconcat`
- [ ] MCP healthcheck + `ltx_generate_sequence` still works (`mcp_server.py`)

## Merge notes

- Requires **ltx-2-mlx v0.14.9** MLX packages (see `requirements.txt` comments).
- `web/dist/` is gitignored — CI/users must `npm run build` after pull.
- No changes to default WebSocket protocol; Web UI uses embedded in-process path for generation.
