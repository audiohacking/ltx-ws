import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { snapshotFromClip } from "./clipEditor";
import { applyProgressEvent } from "./progress";
import type { Clip, Config, ProgressState } from "./types";

const API = "";

async function fetchConfig(): Promise<Config> {
  const r = await fetch(`${API}/api/config`);
  if (!r.ok) throw new Error("Failed to load config");
  return r.json();
}

async function fetchClips(chainId?: string): Promise<Clip[]> {
  const q = chainId ? `?chain_id=${encodeURIComponent(chainId)}` : "";
  const r = await fetch(`${API}/api/clips${q}`);
  if (!r.ok) throw new Error("Failed to load clips");
  const data = await r.json();
  return data.clips as Clip[];
}

function formatBytes(n?: number) {
  if (!n) return "";
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(frames?: number, fps = 24) {
  if (!frames) return "";
  const s = frames / fps;
  return `${s.toFixed(1)}s`;
}

function pickPlaybackClip(clips: Clip[], chainId: string): string | null {
  const chain = clips.filter(
    (c) => c.chain_id === chainId && c.status === "done" && c.video_url,
  );
  const merged = chain.find((c) => c.label === "MERGED");
  const current = chain.find((c) => c.label === "CURRENT");
  const latest = [...chain].sort((a, b) => b.clip_index - a.clip_index)[0];
  return merged?.id ?? current?.id ?? latest?.id ?? null;
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [chainId, setChainId] = useState<string | null>(null);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Options
  const [model, setModel] = useState("auto");
  const [mode, setMode] = useState("generate");
  const [resolutionId, setResolutionId] = useState("704x480");
  const [durationId, setDurationId] = useState("5s");
  const [clipMultiplier, setClipMultiplier] = useState(1);
  const [numSteps, setNumSteps] = useState(8);
  const [seed, setSeed] = useState<string>("");
  const [autocontinue, setAutocontinue] = useState(true);
  const [autoconcat, setAutoconcat] = useState(false);
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string | null>(null);
  const [audioPath, setAudioPath] = useState<string | null>(null);
  const [audioName, setAudioName] = useState<string | null>(null);
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [retakeStart, setRetakeStart] = useState(1);
  const [retakeEnd, setRetakeEnd] = useState(1);
  const [extendFrames, setExtendFrames] = useState(2);
  const [extendDirection, setExtendDirection] = useState("after");
  const [showOptions, setShowOptions] = useState(true);

  const imageRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLInputElement>(null);

  const libraryClips = useMemo(() => {
    return clips
      .filter((c) => c.status === "done" && c.video_url)
      .sort((a, b) => b.created_at.localeCompare(a.created_at))
      .slice(0, 48);
  }, [clips]);

  const chainParts = useMemo(() => {
    if (!chainId || !selectedClipId) return [];
    return clips
      .filter((c) => c.chain_id === chainId && c.status === "done" && c.video_url)
      .sort((a, b) => a.clip_index - b.clip_index);
  }, [clips, chainId, selectedClipId]);

  const showChainPicker = chainParts.length > 1;

  const activeClip = useMemo(() => {
    if (!selectedClipId) return null;
    return clips.find((x) => x.id === selectedClipId) ?? null;
  }, [clips, selectedClipId]);

  const load = useCallback(async () => {
    try {
      const cfg = await fetchConfig();
      setConfig(cfg);
      setModel(
        cfg.embedded && cfg.active_model
          ? cfg.active_model
          : cfg.preferred_model,
      );
      setNumSteps(cfg.defaults.num_steps);
      const all = await fetchClips();
      setClips(all);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (clipMultiplier > 1) {
      setAutocontinue(true);
      setAutoconcat(true);
    }
  }, [clipMultiplier]);

  const resolution = useMemo(() => {
    if (!config) return { width: 704, height: 480 };
    const p = config.resolution_presets.find((r) => r.id === resolutionId);
    return { width: p?.width ?? 704, height: p?.height ?? 480 };
  }, [config, resolutionId]);

  const durationSeconds = useMemo(() => {
    if (!config) return 5;
    const p = config.duration_presets.find((d) => d.id === durationId);
    return p?.seconds ?? 5;
  }, [config, durationId]);

  const totalDurationSeconds = durationSeconds * clipMultiplier;
  const isMultiClip = clipMultiplier > 1;
  const editingChain =
    !isMultiClip &&
    Boolean(chainId && activeClip?.status === "done");

  function applyClipSelection(clip: Clip) {
    const snap = snapshotFromClip(clip, config, {
      numSteps: config?.defaults.num_steps ?? 8,
    });
    setSelectedClipId(clip.id);
    setChainId(clip.chain_id);
    setPrompt(snap.prompt);
    setMode(snap.mode);
    setResolutionId(snap.resolutionId);
    setDurationId(snap.durationId);
    setClipMultiplier(snap.clipMultiplier);
    setNumSteps(snap.numSteps);
    setSeed(snap.seed);
    setAutocontinue(snap.autocontinue);
    setAutoconcat(snap.autoconcat);
    setShowOptions(true);
    setError(null);
  }

  async function deleteGeneration(clip: Clip) {
    const siblings = clips.filter((c) => c.chain_id === clip.chain_id);
    const deleteChain = siblings.length > 1;
    const msg = deleteChain
      ? "Delete this entire generation (all clips in the chain)?"
      : "Delete this video?";
    if (!confirm(msg)) return;

    const url = deleteChain
      ? `${API}/api/chains/${encodeURIComponent(clip.chain_id)}`
      : `${API}/api/clips/${encodeURIComponent(clip.id)}`;
    const r = await fetch(url, { method: "DELETE" });
    if (!r.ok) {
      setError("Could not delete");
      return;
    }
    const all = await fetchClips();
    setClips(all);
    if (selectedClipId === clip.id || deleteChain) {
      startNewProject();
    }
  }

  function startNewProject() {
    setChainId(null);
    setSelectedClipId(null);
    setPrompt("");
    setClipMultiplier(1);
    setError(null);
  }

  const needsImageUpload = mode === "i2v";
  const needsAudioUpload = mode === "a2v";
  const needsVideoUpload = mode === "retake" || mode === "extend";

  async function uploadFile(file: File, kind: string): Promise<string> {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`${API}/api/upload?kind=${kind}`, {
      method: "POST",
      body: fd,
    });
    if (!r.ok) throw new Error(`Upload failed: ${kind}`);
    const data = await r.json();
    return data.path as string;
  }

  async function changeModel(newModel: string, restart: boolean) {
    setModel(newModel);
    const r = await fetch(`${API}/api/config/model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: newModel, restart_server: restart }),
    });
    if (!r.ok) return;
    const data = await r.json();
    setConfig((c) =>
      c ? { ...c, server_connected: data.server_connected } : c,
    );
  }

  async function subscribeRun(runId: string, runChainId: string) {
    let closed = false;
    let autoconcatRun = false;
    const es = new EventSource(`${API}/api/runs/${runId}/events`);

    const finishRun = async () => {
      if (closed) return;
      closed = true;
      es.close();
      setBusy(false);
      setProgress(null);
      setChainId(runChainId);
      const all = await fetchClips(runChainId);
      setClips(all);
      setSelectedClipId(pickPlaybackClip(all, runChainId));
    };

    const setFromProtocol = (e: Record<string, unknown>) => {
      if (e.type === "queue_status") {
        setProgress({
          phase: "queued",
          message: `Queue position ${e.position ?? "?"}`,
        });
      } else if (
        e.type === "generation_keepalive" ||
        e.type === "generation_status_ack"
      ) {
        setProgress((prev) => applyProgressEvent(prev, e));
      } else if (e.type === "gpu_assigned") {
        setProgress({ phase: "generating", message: "GPU assigned — starting…" });
      } else if (e.type === "ltx2_segment_start") {
        setProgress((prev) => ({
          phase: "generating",
          message: "Denoising…",
          ...prev,
        }));
      } else if (e.type === "error") {
        setError(String(e.message || "Generation error"));
      }
    };
    es.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "run_started") {
        autoconcatRun = Boolean(msg.autoconcat);
        setProgress({
          phase: "queued",
          message: autoconcatRun
            ? `Generating ${msg.clip_count ?? "?"} clips (autoconcat)…`
            : "Generation queued…",
        });
      } else if (msg.type === "clip_started") {
        const idx = typeof msg.index === "number" ? msg.index + 1 : "?";
        const total = msg.total_clips ?? "?";
        setProgress({
          phase: "running",
          message:
            autoconcatRun && total !== "?"
              ? `Generating clip ${idx}/${total}…`
              : "Starting clip…",
        });
      } else if (msg.type === "generation_progress") {
        setProgress((prev) => applyProgressEvent(prev, msg));
      } else if (msg.type === "protocol") {
        setFromProtocol(msg.event as Record<string, unknown>);
      } else if (msg.type === "download_progress") {
        setProgress({
          phase: "downloading",
          message: `Receiving video ${msg.kb} KB`,
          kb: msg.kb,
        });
      } else if (msg.type === "clip_done") {
        const idx = typeof msg.index === "number" ? msg.index + 1 : "?";
        const total = msg.total_clips ?? "?";
        if (autoconcatRun) {
          setProgress({
            phase: "clip_done",
            message:
              total !== "?"
                ? `Clip ${idx}/${total} done — ${idx === total ? "merging…" : "continuing…"}`
                : "Clip saved — continuing…",
          });
        } else {
          setProgress({ phase: "clip_done", message: "Clip saved" });
          if (msg.clip_id && msg.video_url) {
            setSelectedClipId(msg.clip_id as string);
            setClips((prev) => {
              const others = prev.filter((c) => c.id !== msg.clip_id);
              const existing = prev.find((c) => c.id === msg.clip_id);
              return [
                ...others,
                {
                  ...(existing ?? {}),
                  id: msg.clip_id as string,
                  video_url: msg.video_url as string,
                  chain_id: runChainId,
                  status: "done",
                  label: existing?.label ?? "CURRENT",
                  prompt: existing?.prompt ?? "",
                  filename: existing?.filename ?? "",
                  clip_index: existing?.clip_index ?? 0,
                  mode: existing?.mode ?? "generate",
                  created_at: existing?.created_at ?? new Date().toISOString(),
                } as Clip,
              ];
            });
          }
        }
      } else if (msg.type === "merged") {
        setProgress({ phase: "merged", message: "Clips merged" });
        const clipId = msg.clip_id as string | undefined;
        const videoUrl = msg.video_url as string | undefined;
        if (clipId && videoUrl) {
          setSelectedClipId(clipId);
          setClips((prev) => {
            const chainPrompt =
              prev.find((c) => c.chain_id === runChainId)?.prompt ?? "Merged clip";
            const others = prev.filter((c) => c.chain_id !== runChainId);
            return [
              ...others,
              {
                id: clipId,
                video_url: videoUrl,
                filename: (msg.filename as string) ?? "",
                chain_id: runChainId,
                label: "MERGED",
                status: "done",
                prompt: chainPrompt,
                clip_index: 0,
                mode: mode,
                created_at: new Date().toISOString(),
              } as Clip,
            ];
          });
        }
        fetchClips(runChainId).then((all) => {
          setClips(all);
          setSelectedClipId(pickPlaybackClip(all, runChainId) ?? clipId ?? null);
        });
      } else if (msg.type === "run_complete" || msg.type === "run_done") {
        finishRun();
      } else if (msg.type === "error" || msg.type === "clip_failed") {
        setError(msg.error || msg.message || "Failed");
        es.close();
        setBusy(false);
      }
    };
    es.onerror = () => {
      if (closed) return;
      closed = true;
      es.close();
      setBusy(false);
      setProgress(null);
      setError((prev) => prev ?? "Lost connection to server while waiting for progress.");
    };
  }

  async function handleGenerate() {
    if (!prompt.trim() || busy) return;
    setError(null);
    setBusy(true);
    setProgress({ phase: "starting", message: "Submitting…" });

    const isChainEdit = editingChain && autocontinue;

    const body: Record<string, unknown> = {
      prompt: prompt.trim(),
      mode,
      width: resolution.width,
      height: resolution.height,
      duration_seconds: durationSeconds,
      clip_count: clipMultiplier,
      num_steps: numSteps,
      autocontinue: autocontinue || isMultiClip,
      autoconcat: autoconcat || isMultiClip,
      chain_id: isChainEdit ? chainId : undefined,
      continue_from: isChainEdit ? activeClip?.id : undefined,
      image_path: imagePath,
      audio_path: audioPath,
      video_path: videoPath,
      retake_start: retakeStart,
      retake_end: retakeEnd,
      extend_frames: extendFrames,
      extend_direction: extendDirection,
    };
    if (seed.trim()) body.seed = parseInt(seed, 10);

    try {
      const r = await fetch(`${API}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        const detail = err.detail;
        const message =
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map((d: { msg?: string }) => d.msg).join("; ")
              : "Generate failed";
        throw new Error(message);
      }
      const data = await r.json();
      setChainId(data.chain_id);
      setSelectedClipId(null);
      setPrompt("");
      setProgress({ phase: "queued", message: "Queued — starting…" });
      subscribeRun(data.run_id, data.chain_id);
      const all = await fetchClips(data.chain_id);
      setClips(all);
    } catch (e) {
      setError(String(e));
      setBusy(false);
      setProgress(null);
    }
  }

  const serverOk = config?.server_connected;

  const endpointLabel = useMemo(() => {
    if (typeof window === "undefined") return config?.server_url ?? "";
    const ws = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws`;
    return ws;
  }, [config?.server_url]);

  const canSubmit = useMemo(() => {
    if (!prompt.trim() || busy || !serverOk) return false;
    const continuing = editingChain && autocontinue;
    if (mode === "i2v" && !imagePath && !continuing) return false;
    if (mode === "a2v" && !audioPath) return false;
    if ((mode === "retake" || mode === "extend") && !videoPath) return false;
    return true;
  }, [
    prompt,
    busy,
    serverOk,
    mode,
    imagePath,
    audioPath,
    videoPath,
    autocontinue,
    activeClip,
    chainId,
  ]);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark">LTX</span>
          <span className="brand-sub">local WebSocket</span>
        </div>
        <div className="header-status">
          <button type="button" className="btn-secondary" onClick={startNewProject}>
            New project
          </button>
          <span
            className={`status-dot ${serverOk ? "ok" : "off"}`}
            title={endpointLabel}
          />
          {serverOk ? "Server connected" : "Server offline"}
        </div>
      </header>

      <div className="app-body">
        <div className="app-main">
        <section className="player-section">
          <div className="player-wrap">
            {activeClip?.video_url ? (
              <video
                className="player"
                src={activeClip.video_url}
                controls
                autoPlay
                loop
                playsInline
              />
            ) : (
              <div className="player placeholder">
                {busy ? progress?.message ?? "Generating…" : "Your video will appear here"}
              </div>
            )}
            {busy && (
              <div className="progress-overlay">
                <div className="progress-bar">
                  {progress?.pct != null ? (
                    <div
                      className="progress-fill"
                      style={{ width: `${Math.min(100, progress.pct)}%` }}
                    />
                  ) : (
                    <div className="progress-pulse" />
                  )}
                </div>
                <span>{progress?.message ?? "Working…"}</span>
              </div>
            )}
          </div>

          {error && <div className="error-banner">{error}</div>}
          {showChainPicker && (
            <div className="chain-picker">
              <label className="chain-picker-label">
                Chain part
                <select
                  className="chain-picker-select"
                  value={selectedClipId ?? ""}
                  onChange={(e) => {
                    const c = chainParts.find((x) => x.id === e.target.value);
                    if (c) applyClipSelection(c);
                  }}
                >
                  {chainParts.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.label}
                      {c.num_frames
                        ? ` · ${formatDuration(c.num_frames, config?.defaults.fps ?? 24)}`
                        : ""}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}
        </section>

        <section className="composer">
          <div className="prompt-row">
            <input
              className="prompt-input"
              placeholder={
                editingChain
                  ? "What do you want to edit?"
                  : "What video do you want to create?"
              }
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleGenerate()}
              disabled={busy}
            />
            <button
              type="button"
              className="btn-generate"
              onClick={handleGenerate}
              disabled={!canSubmit}
            >
              ↑
            </button>
          </div>

          <button
            type="button"
            className="options-toggle"
            onClick={() => setShowOptions((v) => !v)}
          >
            {showOptions ? "Hide options" : "Show options"}
          </button>

          {showOptions && config && (
            <div className="options-panel">
              <div className="options-row">
                <label>
                  Model
                  <select
                    value={model}
                    onChange={(e) => changeModel(e.target.value, false)}
                  >
                    {config.models.map((m) => (
                      <option key={m.id} value={m.repo}>{m.label}</option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => changeModel(model, true)}
                  disabled={config.embedded}
                >
                  Apply model (restart server)
                </button>
              </div>
              <p className="hint">
                {config.embedded && config.active_model
                  ? `Running: ${config.active_model}. `
                  : ""}
                {config.model_note}
              </p>

              <div className="options-grid">
                <label>
                  Mode
                  <select value={mode} onChange={(e) => setMode(e.target.value)}>
                    {config.generation_modes.map((m) => (
                      <option key={m.id} value={m.id}>{m.label}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Resolution
                  <select
                    value={resolutionId}
                    onChange={(e) => setResolutionId(e.target.value)}
                  >
                    {config.resolution_presets.map((r) => (
                      <option key={r.id} value={r.id}>{r.label}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Duration
                  <select
                    value={durationId}
                    onChange={(e) => setDurationId(e.target.value)}
                  >
                    {config.duration_presets.map((d) => (
                      <option key={d.id} value={d.id}>{d.label}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Clips (× duration)
                  <select
                    value={clipMultiplier}
                    onChange={(e) => setClipMultiplier(Number(e.target.value))}
                  >
                    {Array.from(
                      { length: config.clip_multiplier_max ?? 10 },
                      (_, i) => i + 1,
                    ).map((n) => (
                      <option key={n} value={n}>
                        ×{n}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Steps
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={numSteps}
                    onChange={(e) => setNumSteps(Number(e.target.value))}
                  />
                </label>
                <label>
                  Seed
                  <input
                    type="text"
                    placeholder="random"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value)}
                  />
                </label>
              </div>

              {isMultiClip && (
                <p className="hint duration-total">
                  ~{durationSeconds}s × {clipMultiplier} clips ≈ ~{totalDurationSeconds}s
                  merged (autocontinue + autoconcat on)
                </p>
              )}

              <div className="options-checks">
                <label className="check">
                  <input
                    type="checkbox"
                    checked={autocontinue}
                    onChange={(e) => setAutocontinue(e.target.checked)}
                    disabled={isMultiClip}
                  />
                  Autocontinue (last frame → next clip)
                </label>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={autoconcat}
                    onChange={(e) => setAutoconcat(e.target.checked)}
                    disabled={isMultiClip}
                  />
                  Autoconcat (merge clips with ffmpeg)
                </label>
              </div>

              {(needsImageUpload || needsAudioUpload || needsVideoUpload || mode === "generate") && (
                <div className="media-panel">
                  <span className="media-panel-title">Source media</span>
                  {(needsImageUpload || mode === "generate") && (
                    <label className="media-upload">
                      <span className="media-upload-label">
                        {needsImageUpload
                          ? "Source image (required)"
                          : "Start image (optional)"}
                      </span>
                      <input
                        ref={imageRef}
                        type="file"
                        accept="image/*"
                        onChange={async (e) => {
                          const f = e.target.files?.[0];
                          if (f) {
                            setImagePath(await uploadFile(f, "image"));
                            setImageName(f.name);
                          }
                        }}
                      />
                      <span className="media-upload-hint">
                        {imageName ?? "Choose image file…"}
                      </span>
                    </label>
                  )}
                  {needsAudioUpload && (
                    <label className="media-upload">
                      <span className="media-upload-label">
                        Source audio {needsAudioUpload ? "(required)" : ""}
                      </span>
                      <input
                        ref={audioRef}
                        type="file"
                        accept="audio/*"
                        onChange={async (e) => {
                          const f = e.target.files?.[0];
                          if (f) {
                            setAudioPath(await uploadFile(f, "audio"));
                            setAudioName(f.name);
                          }
                        }}
                      />
                      <span className="media-upload-hint">
                        {audioName ?? "Choose audio file…"}
                      </span>
                    </label>
                  )}
                  {needsVideoUpload && (
                    <label className="media-upload">
                      <span className="media-upload-label">Source video (required)</span>
                      <input
                        ref={videoRef}
                        type="file"
                        accept="video/*"
                        onChange={async (e) => {
                          const f = e.target.files?.[0];
                          if (f) setVideoPath(await uploadFile(f, "video"));
                        }}
                      />
                      <span className="media-upload-hint">
                        {videoPath ? "✓ uploaded" : "Choose video file…"}
                      </span>
                    </label>
                  )}
                </div>
              )}

              {(mode === "retake" || mode === "extend") && (
                <div className="options-grid">
                  {mode === "retake" && (
                    <>
                      <label>
                        Retake start
                        <input
                          type="number"
                          value={retakeStart}
                          onChange={(e) => setRetakeStart(Number(e.target.value))}
                        />
                      </label>
                      <label>
                        Retake end
                        <input
                          type="number"
                          value={retakeEnd}
                          onChange={(e) => setRetakeEnd(Number(e.target.value))}
                        />
                      </label>
                    </>
                  )}
                  {mode === "extend" && (
                    <>
                      <label>
                        Extend frames
                        <input
                          type="number"
                          value={extendFrames}
                          onChange={(e) => setExtendFrames(Number(e.target.value))}
                        />
                      </label>
                      <label>
                        Direction
                        <select
                          value={extendDirection}
                          onChange={(e) => setExtendDirection(e.target.value)}
                        >
                          <option value="after">After</option>
                          <option value="before">Before</option>
                        </select>
                      </label>
                    </>
                  )}
                </div>
              )}

              {activeClip && (
                <p className="meta">
                  Viewing: {formatDuration(activeClip.num_frames, config.defaults.fps)}
                  {activeClip.width && activeClip.height
                    ? ` · ${activeClip.width}×${activeClip.height}`
                    : ""}
                  {activeClip.bytes ? ` · ${formatBytes(activeClip.bytes)}` : ""}
                </p>
              )}
            </div>
          )}
        </section>
        </div>

        <aside className="library">
          <div className="library-header">
            <span className="library-title">Library</span>
            <span className="library-count">{libraryClips.length}</span>
          </div>
          <div className="library-grid">
            {libraryClips.map((clip) => (
              <div
                key={clip.id}
                className={`library-card-wrap ${
                  activeClip?.id === clip.id ? "active" : ""
                }`}
              >
                <button
                  type="button"
                  className="library-card"
                  onClick={() => applyClipSelection(clip)}
                  title={clip.prompt}
                >
                  {clip.video_url && (
                    <video
                      className="library-thumb"
                      src={clip.video_url}
                      muted
                      playsInline
                      preload="metadata"
                    />
                  )}
                  <span className={`library-label ${clip.label.toLowerCase()}`}>
                    {clip.label}
                  </span>
                  <span className="library-prompt">{clip.prompt}</span>
                </button>
                <button
                  type="button"
                  className="library-delete"
                  title="Delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteGeneration(clip);
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}
