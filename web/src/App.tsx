import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { snapshotFromClip } from "./clipEditor";
import { applyProgressEvent } from "./progress";
import type { Clip, Config, LoraPreset, ProgressState } from "./types";

const API = "";
const MODEL_PREF_KEY = "ltx-ws-preferred-model";
const LORA_SEL_KEY = "ltx-ws-lora-preset-ids";
const BLOB_VIDEO_PREFIX = "blob:";

type LoraActivity =
  | { phase: "idle" }
  | {
      phase: "working";
      label: string;
      index: number;
      total: number;
      downloading: boolean;
    }
  | { phase: "ready"; message: string }
  | { phase: "error"; message: string };

async function cacheVideoAsBlobUrl(serverUrl: string): Promise<string> {
  const res = await fetch(serverUrl);
  if (!res.ok) throw new Error(`Video download failed (${res.status})`);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

function preserveBlobVideoUrls(prev: Clip[], incoming: Clip[]): Clip[] {
  const blobById = new Map(
    prev
      .filter((c) => c.video_url?.startsWith(BLOB_VIDEO_PREFIX))
      .map((c) => [c.id, c.video_url] as const),
  );
  return incoming.map((c) => {
    const blob = blobById.get(c.id);
    return blob ? { ...c, video_url: blob } : c;
  });
}

function revokeBlobVideoUrls(clips: Clip[]) {
  for (const clip of clips) {
    const url = clip.video_url;
    if (url?.startsWith(BLOB_VIDEO_PREFIX)) {
      URL.revokeObjectURL(url);
    }
  }
}

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

/** Merge server clip lists into local state (by id); used when refreshing one chain. */
function mergeClips(prev: Clip[], incoming: Clip[]): Clip[] {
  const byId = new Map(prev.map((c) => [c.id, c]));
  for (const c of incoming) {
    byId.set(c.id, c);
  }
  return Array.from(byId.values());
}

/** Replace all clips for one chain (e.g. after autoconcat removes fragments). */
function replaceChainClips(prev: Clip[], chainId: string, chainClips: Clip[]): Clip[] {
  const rest = prev.filter((c) => c.chain_id !== chainId);
  return [...rest, ...preserveBlobVideoUrls(prev, chainClips)];
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

function ChainMethodPicker({
  chainMethod,
  onChange,
  className,
}: {
  chainMethod: string;
  onChange: (method: string) => void;
  className?: string;
}) {
  return (
    <div className={`chain-method-panel${className ? ` ${className}` : ""}`}>
      <span className="chain-method-label">Chain</span>
      <div className="chain-method-radios">
        <label className="check chain-method-option">
          <input
            type="radio"
            name="chainMethod"
            value="autocontinue"
            checked={chainMethod === "autocontinue"}
            onChange={() => onChange("autocontinue")}
          />
          Autocontinue
        </label>
        <label className="check chain-method-option">
          <input
            type="radio"
            name="chainMethod"
            value="native_extend"
            checked={chainMethod === "native_extend"}
            onChange={() => onChange("native_extend")}
          />
          Extend video
        </label>
      </div>
    </div>
  );
}

function loraSelectionSummary(presets: LoraPreset[], selectedIds: string[]) {
  const selected = presets.filter((p) => selectedIds.includes(p.id));
  if (selected.length === 0) return "None";
  if (selected.length === 1) {
    const raw = selected[0].label.replace(/\s*\(default\)\s*$/i, "").trim();
    return raw.length > 20 ? `${raw.slice(0, 19)}…` : raw;
  }
  return `${selected.length} LoRAs`;
}

function loraSelectionTitle(presets: LoraPreset[], selectedIds: string[]) {
  const selected = presets.filter((p) => selectedIds.includes(p.id));
  if (!selected.length) return "No LoRA selected";
  return selected.map((p) => p.label).join(", ");
}

function LoraMultiSelect({
  presets,
  selectedIds,
  disabled,
  onToggle,
  onRemoveCustom,
}: {
  presets: LoraPreset[];
  selectedIds: string[];
  disabled?: boolean;
  onToggle: (id: string, checked: boolean) => void;
  onRemoveCustom: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const summary = loraSelectionSummary(presets, selectedIds);
  const title = loraSelectionTitle(presets, selectedIds);

  return (
    <div className={`multi-select${open ? " is-open" : ""}`} ref={rootRef}>
      <button
        type="button"
        className="multi-select-trigger"
        disabled={disabled}
        aria-expanded={open}
        title={title}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="multi-select-trigger-text">{summary}</span>
      </button>
      {open && (
        <div className="multi-select-menu" role="listbox" aria-label="LoRA presets">
          {presets.map((p) => (
            <label key={p.id} className="multi-select-item">
              <input
                type="checkbox"
                checked={selectedIds.includes(p.id)}
                disabled={disabled}
                onChange={(e) => onToggle(p.id, e.target.checked)}
              />
              <span className="multi-select-item-label">{p.label}</span>
              {p.custom ? (
                <button
                  type="button"
                  className="lora-remove"
                  title="Remove"
                  disabled={disabled}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onRemoveCustom(p.id);
                  }}
                >
                  ×
                </button>
              ) : null}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [chainId, setChainId] = useState<string | null>(null);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
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
  const [audiocontinue, setAudiocontinue] = useState(false);
  const [chainMethod, setChainMethod] = useState("autocontinue");
  const [enhancePrompt, setEnhancePrompt] = useState(false);
  const [pipelineProfile, setPipelineProfile] = useState("distilled");
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string | null>(null);
  const [endImagePath, setEndImagePath] = useState<string | null>(null);
  const [endImageName, setEndImageName] = useState<string | null>(null);
  const [audioPath, setAudioPath] = useState<string | null>(null);
  const [audioName, setAudioName] = useState<string | null>(null);
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [retakeStart, setRetakeStart] = useState(1);
  const [retakeEnd, setRetakeEnd] = useState(1);
  const [extendFrames, setExtendFrames] = useState(2);
  const [extendDirection, setExtendDirection] = useState("after");
  const [showOptions, setShowOptions] = useState(true);
  const [loraPresetIds, setLoraPresetIds] = useState<string[]>([]);
  const [loraActivity, setLoraActivity] = useState<LoraActivity>({ phase: "idle" });
  const [loraBusy, setLoraBusy] = useState(false);
  const [customLoraUrl, setCustomLoraUrl] = useState("");
  const [customLoraLabel, setCustomLoraLabel] = useState("");
  const [customLoraScale, setCustomLoraScale] = useState("1.0");
  const [addingCustomLora, setAddingCustomLora] = useState(false);

  const imageRef = useRef<HTMLInputElement>(null);
  const endImageRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLInputElement>(null);
  const ensuredLoraSpecsRef = useRef<Set<string>>(new Set());
  const loraPresetsRef = useRef<LoraPreset[]>([]);

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

  const ensureLoraPresets = useCallback(async (
    presetIds: string[],
    presetsOverride?: LoraPreset[],
  ) => {
    const presets = presetsOverride ?? loraPresetsRef.current;
    const selected = presetIds
      .map((id) => presets.find((p) => p.id === id))
      .filter((p): p is NonNullable<typeof p> => Boolean(p?.spec));
    if (!selected.length) {
      setLoraActivity({ phase: "idle" });
      return;
    }

    const pending = selected.filter((p) => !ensuredLoraSpecsRef.current.has(p.spec));
    if (!pending.length) {
      setLoraActivity({
        phase: "ready",
        message: `${selected.length} LoRA(s) ready`,
      });
      return;
    }

    setLoraBusy(true);
    try {
      for (let i = 0; i < pending.length; i++) {
        const preset = pending[i];
        setLoraActivity({
          phase: "working",
          label: preset.label,
          index: i + 1,
          total: pending.length,
          downloading: true,
        });
        const r = await fetch(`${API}/api/loras/ensure`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ spec: preset.spec }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err.detail || `LoRA download failed: ${preset.label}`);
        }
        const data = (await r.json()) as { cached?: boolean };
        ensuredLoraSpecsRef.current.add(preset.spec);
        if (data.cached) {
          setLoraActivity({
            phase: "working",
            label: preset.label,
            index: i + 1,
            total: pending.length,
            downloading: false,
          });
        }
      }
      setLoraActivity({
        phase: "ready",
        message: `${selected.length} LoRA(s) ready`,
      });
    } catch (e) {
      setLoraActivity({ phase: "error", message: String(e) });
    } finally {
      setLoraBusy(false);
    }
  }, []);

  const load = useCallback(async () => {
    try {
      const cfg = await fetchConfig();
      setConfig(cfg);
      loraPresetsRef.current = cfg.lora_presets ?? [];
      const preferredModel =
        cfg.preferred_model ||
        localStorage.getItem(MODEL_PREF_KEY) ||
        cfg.default_model ||
        "auto";
      setModel(preferredModel);
      localStorage.setItem(MODEL_PREF_KEY, preferredModel);
      setNumSteps(cfg.defaults.num_steps);
      let loraIds: string[] = [];
      try {
        const stored = localStorage.getItem(LORA_SEL_KEY);
        if (stored) {
          const parsed = JSON.parse(stored) as unknown;
          if (Array.isArray(parsed)) {
            loraIds = parsed.map(String).filter(Boolean);
          }
        }
      } catch {
        /* ignore */
      }
      if (!loraIds.length) {
        loraIds =
          cfg.preferred_lora_preset_ids?.length
            ? cfg.preferred_lora_preset_ids
            : cfg.default_lora_preset_id && cfg.default_lora_preset_id !== "none"
              ? [cfg.default_lora_preset_id]
              : [];
      }
      setLoraPresetIds(loraIds);
      if (loraIds.length) {
        void ensureLoraPresets(loraIds, cfg.lora_presets);
      } else {
        setLoraActivity({ phase: "idle" });
      }
      const all = await fetchClips();
      setClips(all);
    } catch (e) {
      setError(String(e));
    }
  }, [ensureLoraPresets]);

  const persistLoraSelection = useCallback(async (ids: string[]) => {
    try {
      localStorage.setItem(LORA_SEL_KEY, JSON.stringify(ids));
    } catch {
      /* ignore */
    }
    try {
      await fetch(`${API}/api/config/loras`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset_ids: ids }),
      });
    } catch (err) {
      console.warn("Could not persist LoRA selection", err);
    }
  }, []);

  const toggleLoraPreset = useCallback(
    (presetId: string, checked: boolean) => {
      if (loraBusy) return;
      setLoraPresetIds((prev) => {
        const next = checked
          ? [...prev.filter((id) => id !== presetId), presetId]
          : prev.filter((id) => id !== presetId);
        void persistLoraSelection(next);
        void ensureLoraPresets(next);
        return next;
      });
    },
    [ensureLoraPresets, loraBusy, persistLoraSelection],
  );

  async function addCustomLora() {
    const spec = customLoraUrl.trim();
    if (!spec || addingCustomLora) return;
    setAddingCustomLora(true);
    setLoraActivity({
      phase: "working",
      label: customLoraLabel.trim() || "Custom LoRA",
      index: 1,
      total: 1,
      downloading: true,
    });
    try {
      const r = await fetch(`${API}/api/loras/custom`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          spec,
          label: customLoraLabel.trim() || undefined,
          scale: parseFloat(customLoraScale) || 1.0,
        }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || "Could not add custom LoRA");
      }
      const data = await r.json();
      const nextPresets = data.lora_presets ?? [];
      loraPresetsRef.current = nextPresets;
      setConfig((c) =>
        c
          ? {
              ...c,
              lora_presets: nextPresets,
              preferred_lora_preset_ids:
                data.preferred_lora_preset_ids ?? c.preferred_lora_preset_ids,
            }
          : c,
      );
      const ids: string[] = data.preferred_lora_preset_ids ?? [];
      setLoraPresetIds(ids);
      setCustomLoraUrl("");
      setCustomLoraLabel("");
      setCustomLoraScale("1.0");
      if (data.id) {
        await ensureLoraPresets([data.id], nextPresets);
      }
    } catch (e) {
      setLoraActivity({ phase: "error", message: String(e) });
    } finally {
      setAddingCustomLora(false);
    }
  }

  async function removeCustomLora(presetId: string) {
    if (!presetId.startsWith("custom_") || loraBusy) return;
    try {
      const r = await fetch(`${API}/api/loras/custom/${encodeURIComponent(presetId)}`, {
        method: "DELETE",
      });
      if (!r.ok) throw new Error("Delete failed");
      const data = await r.json();
      const nextPresets = data.lora_presets ?? [];
      loraPresetsRef.current = nextPresets;
      setConfig((c) =>
        c
          ? {
              ...c,
              lora_presets: nextPresets,
              preferred_lora_preset_ids:
                data.preferred_lora_preset_ids ?? c.preferred_lora_preset_ids,
            }
          : c,
      );
      setLoraPresetIds(data.preferred_lora_preset_ids ?? []);
      setLoraActivity({ phase: "idle" });
    } catch (e) {
      setLoraActivity({ phase: "error", message: String(e) });
    }
  }

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (clipMultiplier > 1) {
      setAutocontinue(true);
      setAutoconcat(true);
    }
  }, [clipMultiplier]);

  useEffect(() => {
    if (audiocontinue) {
      setAutocontinue(true);
      setAutoconcat(true);
    }
  }, [audiocontinue]);

  useEffect(() => {
    if (mode !== "a2v") {
      setAudiocontinue(false);
    }
  }, [mode]);

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
    setAudiocontinue(snap.audiocontinue);
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

  function clearAllMedia() {
    setImagePath(null);
    setImageName(null);
    setEndImagePath(null);
    setEndImageName(null);
    setAudioPath(null);
    setAudioName(null);
    setVideoPath(null);
    if (imageRef.current) imageRef.current.value = "";
    if (endImageRef.current) endImageRef.current.value = "";
    if (audioRef.current) audioRef.current.value = "";
    if (videoRef.current) videoRef.current.value = "";
  }

  function clearMediaForMode(nextMode: string) {
    if (!["i2v", "generate", "a2v", "keyframe"].includes(nextMode)) {
      setImagePath(null);
      setImageName(null);
      if (imageRef.current) imageRef.current.value = "";
    }
    if (nextMode !== "keyframe") {
      setEndImagePath(null);
      setEndImageName(null);
      if (endImageRef.current) endImageRef.current.value = "";
    }
    if (nextMode !== "a2v" && nextMode !== "lipdub") {
      setAudioPath(null);
      setAudioName(null);
      if (audioRef.current) audioRef.current.value = "";
      setAudiocontinue(false);
    }
    if (!["retake", "extend", "lipdub"].includes(nextMode)) {
      setVideoPath(null);
      if (videoRef.current) videoRef.current.value = "";
    }
    if (nextMode === "a2v") {
      setChainMethod("autocontinue");
    }
  }

  async function startNewProject() {
    setClips((prev) => {
      revokeBlobVideoUrls(prev);
      return [];
    });
    setChainId(null);
    setSelectedClipId(null);
    setPrompt("");
    setClipMultiplier(1);
    setAudiocontinue(false);
    setBusy(false);
    setProgress(null);
    setError(null);
    setLoraPresetIds(
      config?.preferred_lora_preset_ids?.length
        ? config.preferred_lora_preset_ids
        : config?.default_lora_preset_id && config.default_lora_preset_id !== "none"
          ? [config.default_lora_preset_id]
          : [],
    );
    clearAllMedia();
    try {
      await fetch(`${API}/api/session/clear`, { method: "POST" });
    } catch (err) {
      console.warn("Session clear failed", err);
    }
  }

  const needsImageUpload = mode === "i2v" || mode === "keyframe";
  const isA2v = mode === "a2v";
  const needsEndImageUpload = mode === "keyframe";
  const needsVideoUpload = mode === "retake" || mode === "extend" || mode === "lipdub";
  const showStartImageOptional = mode === "generate";
  const isT2vLike = mode === "generate" || mode === "i2v";
  const showChainMethodChoice =
    isT2vLike && !audiocontinue && isMultiClip;
  const chainMethodLabel =
    chainMethod === "native_extend" ? "extend video" : "autocontinue";
  const showChainedImageHint =
    isA2v &&
    (autocontinue || isMultiClip || audiocontinue) &&
    Boolean(imagePath);

  async function uploadFile(file: File, kind: string): Promise<string> {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`${API}/api/upload?kind=${kind}`, {
      method: "POST",
      body: fd,
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      const detail = err.detail;
      const message =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d: { msg?: string }) => d.msg).join("; ")
            : `Upload failed: ${kind}`;
      throw new Error(message);
    }
    const data = await r.json();
    return data.path as string;
  }

  async function changeModel(newModel: string, restart: boolean) {
    setModel(newModel);
    localStorage.setItem(MODEL_PREF_KEY, newModel);
    const r = await fetch(`${API}/api/config/model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: newModel, restart_server: restart }),
    });
    if (!r.ok) return;
    const data = await r.json();
    setConfig((c) =>
      c
        ? {
            ...c,
            preferred_model: data.preferred_model ?? newModel,
            server_connected: data.server_connected ?? c.server_connected,
          }
        : c,
    );
  }

  function cacheClipVideoLocally(clipId: string, serverUrl: string) {
    cacheVideoAsBlobUrl(serverUrl)
      .then((blobUrl) => {
        setClips((prev) => {
          const existing = prev.find((c) => c.id === clipId);
          const oldUrl = existing?.video_url;
          if (oldUrl?.startsWith(BLOB_VIDEO_PREFIX)) {
            URL.revokeObjectURL(oldUrl);
          }
          return prev.map((c) =>
            c.id === clipId ? { ...c, video_url: blobUrl } : c,
          );
        });
      })
      .catch((err) => {
        console.warn("Failed to cache clip video locally", err);
      });
  }

  async function cancelActiveRun() {
    if (!activeRunId) return;
    try {
      await fetch(`${API}/api/runs/${activeRunId}/cancel`, { method: "POST" });
      setProgress({ phase: "cancelled", message: "Cancelling…" });
    } catch (err) {
      console.warn("Cancel request failed", err);
    }
  }

  async function subscribeRun(runId: string, runChainId: string) {
    setActiveRunId(runId);
    let closed = false;
    let autoconcatRun = false;
    let audiocontinueRun = false;
    let streamFinalOnly = false;
    const es = new EventSource(`${API}/api/runs/${runId}/events`);

    const finishRun = async () => {
      if (closed) return;
      closed = true;
      es.close();
      setActiveRunId(null);
      setBusy(false);
      setProgress(null);
      setChainId(runChainId);
      const chainClips = await fetchClips(runChainId);
      setClips((prev) => replaceChainClips(prev, runChainId, chainClips));
      setSelectedClipId(pickPlaybackClip(chainClips, runChainId));
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
        audiocontinueRun = Boolean(msg.audiocontinue);
        const clipCount = Number(msg.clip_count ?? 0);
        streamFinalOnly =
          autoconcatRun || (Boolean(msg.autocontinue) && clipCount > 1);
        const chainLabel =
          msg.chain_method === "native_extend" ? "extend video" : "autocontinue";
        setProgress({
          phase: "queued",
          message: audiocontinueRun
            ? `Music video: ${msg.clip_count ?? "?"} clips (audiocontinue)…`
            : autoconcatRun
              ? `Generating ${msg.clip_count ?? "?"} clips (${chainLabel} + autoconcat)…`
              : streamFinalOnly
                ? `Generating ${msg.clip_count ?? "?"} clips (${chainLabel})…`
                : "Generation queued…",
        });
      } else if (msg.type === "clip_started") {
        const idx = typeof msg.index === "number" ? msg.index + 1 : "?";
        const total = msg.total_clips ?? "?";
        setProgress({
          phase: "running",
          message:
            (autoconcatRun || audiocontinueRun) && total !== "?"
              ? audiocontinueRun
                ? `Music video clip ${idx}/${total}…`
                : `Generating clip ${idx}/${total}…`
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
        if (streamFinalOnly) {
          setProgress({
            phase: "clip_done",
            message:
              total !== "?"
                ? autoconcatRun
                  ? `Clip ${idx}/${total} done — ${idx === total ? "merging…" : "continuing…"}`
                  : `Clip ${idx}/${total} done — continuing…`
                : "Clip saved — continuing…",
          });
          // Autoconcat: video arrives on `merged`. Autocontinue: only the last clip has video_url.
          if (msg.clip_id && msg.video_url) {
            const clipId = msg.clip_id as string;
            const serverUrl = msg.video_url as string;
            setSelectedClipId(clipId);
            setClips((prev) => {
              const others = prev.filter((c) => c.id !== clipId);
              const existing = prev.find((c) => c.id === clipId);
              return [
                ...others,
                {
                  ...(existing ?? {}),
                  id: clipId,
                  video_url: serverUrl,
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
            cacheClipVideoLocally(clipId, serverUrl);
          }
        } else if (msg.clip_id && msg.video_url) {
          setProgress({ phase: "clip_done", message: "Clip saved" });
          const clipId = msg.clip_id as string;
          const serverUrl = msg.video_url as string;
          setSelectedClipId(clipId);
          setClips((prev) => {
            const others = prev.filter((c) => c.id !== clipId);
            const existing = prev.find((c) => c.id === clipId);
            return [
              ...others,
              {
                ...(existing ?? {}),
                id: clipId,
                video_url: serverUrl,
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
          cacheClipVideoLocally(clipId, serverUrl);
        } else {
          setProgress({ phase: "clip_done", message: "Clip saved" });
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
          cacheClipVideoLocally(clipId, videoUrl);
        }
        fetchClips(runChainId).then((chainClips) => {
          setClips((prev) => replaceChainClips(prev, runChainId, chainClips));
          setSelectedClipId(pickPlaybackClip(chainClips, runChainId) ?? clipId ?? null);
        });
      } else if (msg.type === "run_cancelled") {
        setProgress({
          phase: "cancelled",
          message: String(msg.message || "Generation cancelled"),
        });
        finishRun();
      } else if (msg.type === "run_complete" || msg.type === "run_done") {
        finishRun();
      } else if (msg.type === "error" || msg.type === "clip_failed") {
        setError(msg.error || msg.message || "Failed");
        es.close();
        setActiveRunId(null);
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

    const durationPreset = config?.duration_presets.find((d) => d.id === durationId);

    const body: Record<string, unknown> = {
      prompt: prompt.trim(),
      mode,
      width: resolution.width,
      height: resolution.height,
      duration_seconds: durationSeconds,
      num_frames: durationPreset?.num_frames,
      clip_count: clipMultiplier,
      num_steps: numSteps,
      autocontinue: autocontinue || isMultiClip || audiocontinue,
      autoconcat: autoconcat || isMultiClip || audiocontinue,
      audiocontinue: audiocontinue && mode === "a2v",
      chain_method: chainMethod,
      enhance_prompt: enhancePrompt,
      pipeline_profile: pipelineProfile,
      chain_id: isChainEdit ? chainId : undefined,
      continue_from: isChainEdit ? activeClip?.id : undefined,
    };
    if (mode === "retake") {
      body.retake_start = retakeStart;
      body.retake_end = retakeEnd;
    }
    if (mode === "extend") {
      body.extend_frames = extendFrames;
      body.extend_direction = extendDirection;
    }
    if (
      (mode === "i2v" || mode === "generate" || mode === "a2v" || mode === "keyframe") &&
      imagePath
    ) {
      body.image_path = imagePath;
    }
    if (mode === "keyframe" && endImagePath) {
      body.end_image_path = endImagePath;
    }
    if ((mode === "a2v" || mode === "lipdub") && audioPath) {
      body.audio_path = audioPath;
    }
    if ((mode === "retake" || mode === "extend" || mode === "lipdub") && videoPath) {
      body.video_path = videoPath;
    }
    if (seed.trim()) {
      body.seed = parseInt(seed, 10);
    } else {
      body.seed = -1;
    }

    const selectedLoras = (config?.lora_presets ?? []).filter(
      (p) => loraPresetIds.includes(p.id) && p.spec,
    );
    if (mode === "lipdub" && selectedLoras.length !== 1) {
      setError("LipDub requires exactly one LoRA — select a single preset.");
      setBusy(false);
      setProgress(null);
      return;
    }
    if (selectedLoras.length) {
      body.lora_specs = selectedLoras.map((p) => [p.spec, p.scale]);
    }

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
      const chainClips = await fetchClips(data.chain_id);
      setClips((prev) => mergeClips(prev, chainClips));
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
    if (audiocontinue && !config?.ffmpeg_available) return false;
    if ((mode === "retake" || mode === "extend") && !videoPath) return false;
    return true;
  }, [
    prompt,
    busy,
    serverOk,
    mode,
    imagePath,
    audioPath,
    audiocontinue,
    clipMultiplier,
    config?.ffmpeg_available,
    videoPath,
    autocontinue,
    activeClip,
    chainId,
  ]);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark">LTX-WS</span>
          <span className="brand-sub">Videofentanyl</span>
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
                <div className="progress-overlay-row">
                  <span>{progress?.message ?? "Working…"}</span>
                  {activeRunId && progress?.phase !== "cancelled" && (
                    <button
                      type="button"
                      className="btn-cancel"
                      onClick={() => void cancelActiveRun()}
                    >
                      Cancel
                    </button>
                  )}
                </div>
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
                      <option key={m.id} value={m.id}>{m.label}</option>
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

              <div className="options-grid options-grid-compact">
                <label className="opt-mode">
                  Mode
                  <select
                    value={mode}
                    onChange={(e) => {
                      const next = e.target.value;
                      setMode(next);
                      clearMediaForMode(next);
                    }}
                  >
                    {config.generation_modes.map((m) => (
                      <option key={m.id} value={m.id}>{m.label}</option>
                    ))}
                  </select>
                </label>
                <label className="opt-resolution">
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
                <label className="opt-narrow">
                  Duration
                  <select
                    value={durationId}
                    onChange={(e) => setDurationId(e.target.value)}
                  >
                    {config.duration_presets.map((d) => (
                      <option key={d.id} value={d.id} title={d.label}>
                        {d.label.includes("(test)")
                          ? `~${d.seconds}s*`
                          : `~${d.seconds}s`}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="opt-narrow">
                  Clips
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
                <label className="opt-narrow">
                  Steps
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={numSteps}
                    onChange={(e) => setNumSteps(Number(e.target.value))}
                  />
                </label>
                <label className="opt-seed">
                  Seed
                  <input
                    type="text"
                    placeholder="random"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value)}
                  />
                </label>
              </div>

              {(loraActivity.phase === "working" || loraActivity.phase === "error") && (
                <div
                  className={`lora-status-banner ${
                    loraActivity.phase === "error" ? "error" : "working"
                  }`}
                  role="status"
                  aria-live="polite"
                >
                  {loraActivity.phase === "working" && (
                    <>
                      <span className="lora-status-spinner" aria-hidden />
                      <span>
                        {loraActivity.downloading
                          ? `Downloading: ${loraActivity.label}`
                          : `Verifying: ${loraActivity.label}`}
                        {loraActivity.total > 1
                          ? ` (${loraActivity.index}/${loraActivity.total})`
                          : ""}
                      </span>
                    </>
                  )}
                  {loraActivity.phase === "error" && (
                    <span>{loraActivity.message}</span>
                  )}
                </div>
              )}

              <div className="lora-row">
                <label className="lora-row-select">
                  LoRA
                  <LoraMultiSelect
                    presets={(config.lora_presets ?? []).filter((p) => p.id !== "none")}
                    selectedIds={loraPresetIds}
                    disabled={loraBusy || addingCustomLora}
                    onToggle={(id, checked) => toggleLoraPreset(id, checked)}
                    onRemoveCustom={(id) => void removeCustomLora(id)}
                  />
                </label>
                <div className="lora-row-add">
                  <input
                    type="text"
                    className="lora-add-url"
                    placeholder="URL or path"
                    aria-label="LoRA URL or file path"
                    value={customLoraUrl}
                    disabled={addingCustomLora}
                    onChange={(e) => setCustomLoraUrl(e.target.value)}
                  />
                  <input
                    type="text"
                    className="lora-add-name"
                    placeholder="Label"
                    aria-label="LoRA display name"
                    value={customLoraLabel}
                    disabled={addingCustomLora}
                    onChange={(e) => setCustomLoraLabel(e.target.value)}
                  />
                  <input
                    type="number"
                    className="lora-add-scale"
                    min={0}
                    max={2}
                    step={0.05}
                    aria-label="LoRA strength"
                    title="Strength (0–2)"
                    placeholder="1.0"
                    value={customLoraScale}
                    disabled={addingCustomLora}
                    onChange={(e) => setCustomLoraScale(e.target.value)}
                  />
                  <button
                    type="button"
                    className="btn-secondary btn-compact lora-add-btn"
                    disabled={!customLoraUrl.trim() || addingCustomLora || loraBusy}
                    onClick={() => void addCustomLora()}
                  >
                    {addingCustomLora ? "…" : "Add"}
                  </button>
                </div>
              </div>

              {isMultiClip && !audiocontinue && (
                <p className="hint hint-inline">
                  ~{totalDurationSeconds}s total · {chainMethodLabel}
                </p>
              )}

              {showChainMethodChoice && (
                <ChainMethodPicker
                  chainMethod={chainMethod}
                  onChange={setChainMethod}
                />
              )}

              <div className="options-checks">
                {mode === "a2v" && (
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={audiocontinue}
                      onChange={(e) => {
                        const on = e.target.checked;
                        setAudiocontinue(on);
                        if (on && clipMultiplier < 2) {
                          setClipMultiplier(2);
                        }
                      }}
                      disabled={!audioPath || !config?.ffmpeg_available}
                    />
                    Audiocontinue
                  </label>
                )}
                {mode === "a2v" && !config?.ffmpeg_available && (
                  <p className="hint hint-inline">Requires ffmpeg.</p>
                )}
                <label className="check">
                  <input
                    type="checkbox"
                    checked={enhancePrompt}
                    onChange={(e) => setEnhancePrompt(e.target.checked)}
                  />
                  Enhance prompt
                </label>
                {!isMultiClip && !audiocontinue && (
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={autocontinue}
                      onChange={(e) => setAutocontinue(e.target.checked)}
                    />
                    Chain clips
                  </label>
                )}
                <label className="check">
                  <input
                    type="checkbox"
                    checked={autoconcat}
                    onChange={(e) => setAutoconcat(e.target.checked)}
                    disabled={isMultiClip || audiocontinue}
                  />
                  Autoconcat
                </label>
                <label className="opt-profile">
                  Profile
                  <select
                    value={pipelineProfile}
                    onChange={(e) => setPipelineProfile(e.target.value)}
                  >
                    {(config?.pipeline_profiles ?? [
                      { id: "distilled", label: "Distilled" },
                      { id: "two_stage", label: "Two-stage" },
                      { id: "hq", label: "HQ" },
                      { id: "one_stage", label: "One-stage" },
                    ]).map((p) => (
                      <option key={p.id} value={p.id}>{p.label}</option>
                    ))}
                  </select>
                </label>
              </div>

              {mode === "extend" && (
                <div className="options-grid">
                  <label>
                    Extend frames (latent)
                    <input
                      type="number"
                      min={1}
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
                </div>
              )}

              {(isA2v || needsImageUpload || showStartImageOptional || needsVideoUpload || needsEndImageUpload) && (
                <div className="media-panel">
                  {isA2v && (
                    <>
                      <span className="media-panel-title">Audio to video inputs</span>
                      <div className="media-upload-row">
                        <label className="media-upload">
                          <span className="media-upload-label">
                            Start image (optional, clip 1 only)
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
                        <label className="media-upload">
                          <span className="media-upload-label">
                            Source audio (required)
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
                      </div>
                      {showChainedImageHint && chainMethod === "autocontinue" && (
                        <p className="hint">
                          With autocontinue / audiocontinue, the start image is used for
                          clip 1 only; later clips use the last frame of the prior clip.
                        </p>
                      )}
                    </>
                  )}
                  {!isA2v && (needsImageUpload || showStartImageOptional) && (
                    <>
                      <span className="media-panel-title">Source media</span>
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
                    {needsEndImageUpload && (
                      <label className="media-upload">
                        <span className="media-upload-label">End image (required)</span>
                        <input
                          ref={endImageRef}
                          type="file"
                          accept="image/*"
                          onChange={async (e) => {
                            const f = e.target.files?.[0];
                            if (f) {
                              setEndImagePath(await uploadFile(f, "image"));
                              setEndImageName(f.name);
                            }
                          }}
                        />
                        <span className="media-upload-hint">
                          {endImageName ?? "Choose end frame…"}
                        </span>
                      </label>
                    )}
                    </>
                  )}
                  {needsVideoUpload && (
                    <>
                      {!isA2v && (
                        <span className="media-panel-title">Source media</span>
                      )}
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
                    </>
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
