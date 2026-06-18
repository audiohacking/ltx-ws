import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Config, TrainHealth, TrainJob, TrainPreset } from "./types";
import {
  cancelTrainJob,
  createTrainJob,
  fetchTrainHealth,
  fetchTrainJobs,
  fetchTrainPresets,
  registerTrainedLora,
  resumeTrainJob,
  subscribeTrainJob,
  type TrainManifest,
} from "./api/train";

type WizardStep = "dataset" | "preprocess" | "train" | "runs";

const STEPS: { id: WizardStep; label: string; hint: string }[] = [
  { id: "dataset", label: "Dataset", hint: "Videos & captions" },
  { id: "preprocess", label: "Preprocess", hint: "Resolution & frames" },
  { id: "train", label: "Train", hint: "Hyperparameters" },
  { id: "runs", label: "Runs", hint: "Progress & output" },
];

function formatEta(seconds?: number): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

function phaseLabel(phase?: string): string {
  switch (phase) {
    case "slicing":
      return "Slicing videos";
    case "preprocessing":
      return "Preprocessing latents";
    case "training":
      return "Training LoRA";
    case "done":
      return "Complete";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Cancelled";
    case "interrupted":
      return "Interrupted";
    default:
      return phase || "Queued";
  }
}

export default function TrainPage() {
  const [step, setStep] = useState<WizardStep>("dataset");
  const [health, setHealth] = useState<TrainHealth | null>(null);
  const [presets, setPresets] = useState<TrainPreset[]>([]);
  const [config, setConfig] = useState<Config | null>(null);
  const [jobs, setJobs] = useState<TrainJob[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [registering, setRegistering] = useState(false);
  const [resuming, setResuming] = useState(false);

  const [name, setName] = useState("My LoRA");
  const [preset, setPreset] = useState("t2v");
  const [videos, setVideos] = useState<File[]>([]);
  const [references, setReferences] = useState<File[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [refDragOver, setRefDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const refInputRef = useRef<HTMLInputElement>(null);

  const [sliceEnabled, setSliceEnabled] = useState(false);
  const [sliceInterval, setSliceInterval] = useState(4);
  const [sliceRes, setSliceRes] = useState("384x384");
  const [sliceFps, setSliceFps] = useState(24);
  const [sliceFit, setSliceFit] = useState("crop");
  const [captionTemplate, setCaptionTemplate] = useState("");

  const [width, setWidth] = useState(704);
  const [height, setHeight] = useState(480);
  const [maxFrames, setMaxFrames] = useState(97);
  const [withAudio, setWithAudio] = useState(false);
  const [referenceDownscale, setReferenceDownscale] = useState(2);

  const [steps, setSteps] = useState(2000);
  const [rank, setRank] = useState(64);
  const [learningRate, setLearningRate] = useState(0.0005);
  const [validationPrompts, setValidationPrompts] = useState(
    "a cinematic landscape at sunset\na person walking through neon rain",
  );
  const [validationInterval, setValidationInterval] = useState(500);
  const [checkpointInterval, setCheckpointInterval] = useState(500);
  const [lowRam, setLowRam] = useState(false);
  const [seed, setSeed] = useState(42);

  const activeJob = useMemo(
    () => jobs.find((j) => j.id === activeJobId) ?? null,
    [jobs, activeJobId],
  );

  const selectedPreset = useMemo(
    () => presets.find((p) => p.id === preset) ?? presets[0],
    [presets, preset],
  );

  const refreshJobs = useCallback(async () => {
    try {
      const list = await fetchTrainJobs();
      setJobs(list);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    fetch("/api/config")
      .then((r) => r.json())
      .then((c: Config) => setConfig(c))
      .catch(() => {});
    fetchTrainHealth().then(setHealth).catch(() => {});
    fetchTrainPresets().then(setPresets).catch(() => {});
    refreshJobs();
  }, [refreshJobs]);

  const isV2v = preset === "v2v";

  useEffect(() => {
    if (!selectedPreset) return;
    setWithAudio(selectedPreset.with_audio);
    setLowRam(selectedPreset.low_ram_default);
    if (selectedPreset.id === "v2v") {
      setSteps(3000);
      setValidationPrompts("a person walking in a park");
    }
  }, [selectedPreset?.id]);

  const manifest = useMemo((): TrainManifest => {
    const prompts = validationPrompts
      .split("\n")
      .map((p) => p.trim())
      .filter(Boolean);
    return {
      name,
      preset,
      model_id: config?.preferred_model || config?.active_model || "auto",
      slice: {
        enabled: sliceEnabled,
        interval: sliceInterval,
        res: sliceRes,
        fps: sliceFps,
        fit: sliceFit,
        caption_template: captionTemplate.trim() || undefined,
      },
      preprocess: {
        width,
        height,
        max_frames: maxFrames,
        with_audio: withAudio,
        frame_rate: 24,
        reference_downscale_factor: referenceDownscale,
      },
      train: {
        steps,
        rank,
        learning_rate: learningRate,
        validation_prompts: prompts.length ? prompts : ["a cinematic landscape at sunset"],
        validation_interval: validationInterval,
        checkpoint_interval: checkpointInterval,
        low_ram: lowRam,
        seed,
      },
    };
  }, [
    name,
    preset,
    config,
    sliceEnabled,
    sliceInterval,
    sliceRes,
    sliceFps,
    sliceFit,
    captionTemplate,
    width,
    height,
    maxFrames,
    withAudio,
    referenceDownscale,
    steps,
    rank,
    learningRate,
    validationPrompts,
    validationInterval,
    checkpointInterval,
    lowRam,
    seed,
  ]);

  const updateJob = useCallback((jobId: string, patch: Partial<TrainJob>) => {
    setJobs((prev) =>
      prev.map((j) => (j.id === jobId ? { ...j, ...patch } : j)),
    );
  }, []);

  useEffect(() => {
    if (!activeJobId) return;
    const unsub = subscribeTrainJob(activeJobId, (event) => {
      const type = String(event.type || "");
      if (type === "phase_started") {
        updateJob(activeJobId, { phase: String(event.phase || ""), status: "running" });
      } else if (type === "train_step") {
        updateJob(activeJobId, {
          step: Number(event.step) || 0,
          total_steps: Number(event.total_steps) || 0,
          loss: event.loss != null ? Number(event.loss) : undefined,
          lr: event.lr != null ? Number(event.lr) : undefined,
          eta_s: event.eta_s != null ? Number(event.eta_s) : undefined,
          phase: "training",
          status: "running",
        });
      } else if (type === "train_validation") {
        const videos = (event.videos as TrainJob["validation_clips"]) || [];
        setJobs((prev) =>
          prev.map((j) =>
            j.id === activeJobId
              ? { ...j, validation_clips: [...(j.validation_clips || []), ...videos] }
              : j,
          ),
        );
      } else if (type === "job_done") {
        updateJob(activeJobId, {
          status: "done",
          phase: "done",
          artifact_url: String(event.artifact_url || ""),
          artifact_name: String(event.artifact_name || ""),
        });
      } else if (type === "error") {
        updateJob(activeJobId, {
          status: event.message === "Cancelled" ? "cancelled" : "failed",
          phase: event.message === "Cancelled" ? "cancelled" : "failed",
          error: String(event.message || "Error"),
        });
      } else if (type === "snapshot" && event.job) {
        const snap = event.job as TrainJob;
        updateJob(activeJobId, snap);
      }
      if (type === "job_complete") {
        refreshJobs();
      }
    });
    return unsub;
  }, [activeJobId, updateJob, refreshJobs]);

  function addTargetFiles(fileList: FileList | File[]) {
    const incoming = Array.from(fileList).filter((f) =>
      /\.(mp4|mov|avi|mkv|webm|txt)$/i.test(f.name),
    );
    if (!incoming.length) return;
    setVideos((prev) => {
      const names = new Set(prev.map((f) => f.name));
      const merged = [...prev];
      for (const f of incoming) {
        if (!names.has(f.name)) merged.push(f);
      }
      return merged;
    });
  }

  function addReferenceFiles(fileList: FileList | File[]) {
    const incoming = Array.from(fileList).filter((f) =>
      /\.(mp4|mov|avi|mkv|webm)$/i.test(f.name),
    );
    if (!incoming.length) return;
    setReferences((prev) => {
      const names = new Set(prev.map((f) => f.name));
      const merged = [...prev];
      for (const f of incoming) {
        if (!names.has(f.name)) merged.push(f);
      }
      return merged;
    });
  }

  async function startTraining() {
    setError(null);
    const videoFiles = videos.filter((f) => !f.name.toLowerCase().endsWith(".txt"));
    if (!videoFiles.length) {
      setError("Add at least one target video file (.mp4, .mov, …).");
      setStep("dataset");
      return;
    }
    if (isV2v && !references.length) {
      setError("IC-LoRA requires reference videos paired by matching filename.");
      setStep("dataset");
      return;
    }
    if (!health?.trainer_installed) {
      setError("Install ltx-trainer-mlx on the server (see install hint below).");
      return;
    }
    if (health.generation_active) {
      setError("Wait for the current generation to finish before training.");
      return;
    }
    setSubmitting(true);
    try {
      const captionFiles = videos.filter((f) => f.name.toLowerCase().endsWith(".txt"));
      const result = await createTrainJob(manifest, [...videoFiles, ...captionFiles], references);
      const job: TrainJob = {
        id: result.job_id,
        name: result.name,
        preset: result.preset,
        status: "queued",
        phase: "queued",
        created_at: new Date().toISOString(),
        total_steps: steps,
        validation_clips: [],
      };
      setJobs((prev) => [job, ...prev]);
      setActiveJobId(result.job_id);
      setStep("runs");
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleResume(jobId?: string) {
    const id = jobId || activeJobId;
    if (!id) return;
    setResuming(true);
    setError(null);
    try {
      await resumeTrainJob(id);
      setActiveJobId(id);
      setStep("runs");
      await refreshJobs();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setResuming(false);
    }
  }

  async function handleCancel() {
    if (!activeJobId) return;
    try {
      await cancelTrainJob(activeJobId);
      updateJob(activeJobId, { status: "cancelled", phase: "cancelled" });
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }

  async function handleRegister() {
    if (!activeJobId || !activeJob) return;
    setRegistering(true);
    setError(null);
    try {
      const result = await registerTrainedLora(activeJobId, name, 1.0);
      updateJob(activeJobId, { registered_lora_id: result.id });
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setRegistering(false);
    }
  }

  const trainProgress =
    activeJob?.total_steps && activeJob.total_steps > 0
      ? Math.min(100, ((activeJob.step || 0) / activeJob.total_steps) * 100)
      : 0;

  const videoCount = videos.filter((f) => !f.name.toLowerCase().endsWith(".txt")).length;
  const captionCount = videos.filter((f) => f.name.toLowerCase().endsWith(".txt")).length;
  const referenceCount = references.length;

  return (
    <div className="train-page">
      <div className="train-hero">
        <div>
          <h1 className="train-title">Train a LoRA</h1>
          <p className="train-subtitle">
            Upload clips, preprocess latents, and fine-tune a style LoRA for generation — all on your Mac.
          </p>
        </div>
        <div className="train-health">
          <span
            className={`status-dot ${health?.trainer_installed ? "ok" : "off"}`}
            title={health?.trainer_installed ? "Trainer ready" : "Trainer not installed"}
          />
          <span>{health?.trainer_installed ? "Trainer ready" : "Trainer not installed"}</span>
          {health?.training_active && <span className="train-badge">Training</span>}
          {health?.generation_active && <span className="train-badge warn">Gen active</span>}
        </div>
      </div>

      {!health?.trainer_installed && health?.install_hint && (
        <div className="train-callout">
          <strong>Install training support</strong>
          <code className="train-install-hint">{health.install_hint}</code>
        </div>
      )}

      {error && (
        <div className="train-error" role="alert">
          {error}
        </div>
      )}

      <div className="train-wizard">
        <nav className="train-steps" aria-label="Training wizard">
          {STEPS.map((s, i) => (
            <button
              key={s.id}
              type="button"
              className={`train-step-tab ${step === s.id ? "active" : ""}`}
              onClick={() => setStep(s.id)}
            >
              <span className="train-step-num">{i + 1}</span>
              <span className="train-step-text">
                <span className="train-step-label">{s.label}</span>
                <span className="train-step-hint">{s.hint}</span>
              </span>
            </button>
          ))}
        </nav>

        <div className="train-panel">
          {step === "dataset" && (
            <section className="train-section">
              <h2>Dataset</h2>
              <p className="train-section-lead">
                {isV2v
                  ? "Upload target clips (desired output) and reference clips (conditioning). Pair by matching filename — e.g. scene01.mp4 + scene01.mp4."
                  : "Drop training videos here. Optional .txt caption files with matching names are used when slicing is off."}
              </p>

              <label className="field">
                <span>LoRA name</span>
                <input value={name} onChange={(e) => setName(e.target.value)} placeholder="My style LoRA" />
              </label>

              <div className="preset-grid">
                {presets.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    className={`preset-card ${preset === p.id ? "selected" : ""}`}
                    onClick={() => setPreset(p.id)}
                  >
                    <span className="preset-card-title">{p.label}</span>
                    <span className="preset-card-desc">{p.description}</span>
                    <span className="preset-card-meta">{p.ram_hint}</span>
                  </button>
                ))}
              </div>

              <div
                className={`drop-zone ${dragOver ? "drag-over" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  if (e.dataTransfer.files.length) addTargetFiles(e.dataTransfer.files);
                }}
                onClick={() => fileInputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
                }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept="video/*,.txt"
                  hidden
                  onChange={(e) => e.target.files && addTargetFiles(e.target.files)}
                />
                <div className="drop-zone-icon">↑</div>
                <div className="drop-zone-title">
                  {isV2v ? "Drop target videos" : "Drop videos or click to browse"}
                </div>
                <div className="drop-zone-meta">
                  {videoCount} target video{videoCount !== 1 ? "s" : ""}
                  {captionCount > 0 ? ` · ${captionCount} caption file${captionCount !== 1 ? "s" : ""}` : ""}
                </div>
              </div>

              {isV2v && (
                <>
                  <p className="pairing-note">
                    Reference clips drive IC-LoRA conditioning (e.g. depth maps, edges, or a source take). Use the same
                    filenames as targets so pairing survives slicing.
                  </p>
                  <div
                    className={`drop-zone reference-zone ${refDragOver ? "drag-over" : ""}`}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setRefDragOver(true);
                    }}
                    onDragLeave={() => setRefDragOver(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setRefDragOver(false);
                      if (e.dataTransfer.files.length) addReferenceFiles(e.dataTransfer.files);
                    }}
                    onClick={() => refInputRef.current?.click()}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") refInputRef.current?.click();
                    }}
                  >
                    <input
                      ref={refInputRef}
                      type="file"
                      multiple
                      accept="video/*"
                      hidden
                      onChange={(e) => e.target.files && addReferenceFiles(e.target.files)}
                    />
                    <div className="drop-zone-icon">◎</div>
                    <div className="drop-zone-title">Drop reference videos</div>
                    <div className="drop-zone-meta">
                      {referenceCount} reference video{referenceCount !== 1 ? "s" : ""}
                    </div>
                  </div>
                </>
              )}

              {videos.length > 0 && (
                <ul className="file-list">
                  {videos.map((f) => (
                    <li key={`t-${f.name}`}>
                      <span>{f.name}</span>
                      <span className="file-size">{(f.size / 1024 / 1024).toFixed(1)} MB</span>
                      <button
                        type="button"
                        className="file-remove"
                        aria-label={`Remove ${f.name}`}
                        onClick={() => setVideos((prev) => prev.filter((x) => x !== f))}
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              {isV2v && references.length > 0 && (
                <ul className="file-list">
                  {references.map((f) => (
                    <li key={`r-${f.name}`}>
                      <span>ref: {f.name}</span>
                      <span className="file-size">{(f.size / 1024 / 1024).toFixed(1)} MB</span>
                      <button
                        type="button"
                        className="file-remove"
                        aria-label={`Remove ${f.name}`}
                        onClick={() => setReferences((prev) => prev.filter((x) => x !== f))}
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              <details className="train-advanced">
                <summary>Auto-slice long videos</summary>
                <div className="train-advanced-body">
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={sliceEnabled}
                      onChange={(e) => setSliceEnabled(e.target.checked)}
                    />
                    Slice uploads into fixed-length clips (requires ffmpeg)
                  </label>
                  {sliceEnabled && (
                    <div className="field-grid">
                      <label className="field">
                        <span>Interval (seconds)</span>
                        <input
                          type="number"
                          min={1}
                          step={0.5}
                          value={sliceInterval}
                          onChange={(e) => setSliceInterval(Number(e.target.value))}
                        />
                      </label>
                      <label className="field">
                        <span>Clip resolution</span>
                        <input value={sliceRes} onChange={(e) => setSliceRes(e.target.value)} placeholder="384x384" />
                      </label>
                      <label className="field">
                        <span>FPS</span>
                        <input
                          type="number"
                          min={1}
                          value={sliceFps}
                          onChange={(e) => setSliceFps(Number(e.target.value))}
                        />
                      </label>
                      <label className="field">
                        <span>Fit</span>
                        <select value={sliceFit} onChange={(e) => setSliceFit(e.target.value)}>
                          <option value="crop">crop</option>
                          <option value="pad">pad</option>
                          <option value="stretch">stretch</option>
                        </select>
                      </label>
                      <label className="field span-2">
                        <span>Caption template (optional)</span>
                        <input
                          value={captionTemplate}
                          onChange={(e) => setCaptionTemplate(e.target.value)}
                          placeholder="a video of {filename}"
                        />
                      </label>
                    </div>
                  )}
                </div>
              </details>

              <div className="train-actions">
                <button type="button" className="btn-primary" onClick={() => setStep("preprocess")}>
                  Continue
                </button>
              </div>
            </section>
          )}

          {step === "preprocess" && (
            <section className="train-section">
              <h2>Preprocess</h2>
              <p className="train-section-lead">
                Latents are encoded at this resolution. Frames are rounded to valid LTX lengths (8k+1).
                {isV2v && " Reference latents are encoded at a lower resolution for IC-LoRA conditioning."}
              </p>
              <div className="field-grid">
                <label className="field">
                  <span>Width</span>
                  <input type="number" step={32} value={width} onChange={(e) => setWidth(Number(e.target.value))} />
                </label>
                <label className="field">
                  <span>Height</span>
                  <input type="number" step={32} value={height} onChange={(e) => setHeight(Number(e.target.value))} />
                </label>
                <label className="field">
                  <span>Max frames</span>
                  <input
                    type="number"
                    step={8}
                    value={maxFrames}
                    onChange={(e) => setMaxFrames(Number(e.target.value))}
                  />
                </label>
                <label className="checkbox-row span-2">
                  <input
                    type="checkbox"
                    checked={withAudio}
                    disabled={selectedPreset?.with_audio || isV2v}
                    onChange={(e) => setWithAudio(e.target.checked)}
                  />
                  Encode audio latents
                  {selectedPreset?.with_audio && <span className="field-note"> (required for AV preset)</span>}
                </label>
                {isV2v && (
                  <label className="field">
                    <span>Reference downscale factor</span>
                    <input
                      type="number"
                      min={1}
                      max={8}
                      value={referenceDownscale}
                      onChange={(e) => setReferenceDownscale(Number(e.target.value))}
                    />
                  </label>
                )}
              </div>
              <div className="train-actions">
                <button type="button" className="btn-secondary" onClick={() => setStep("dataset")}>
                  Back
                </button>
                <button type="button" className="btn-primary" onClick={() => setStep("train")}>
                  Continue
                </button>
              </div>
            </section>
          )}

          {step === "train" && (
            <section className="train-section">
              <h2>Training</h2>
              <p className="train-section-lead">
                Typical runs use 1k–3k steps. Validation clips appear during training so you can judge quality early.
              </p>
              <div className="field-grid">
                <label className="field">
                  <span>Steps</span>
                  <input type="number" min={100} step={100} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
                </label>
                <label className="field">
                  <span>LoRA rank</span>
                  <input type="number" min={8} step={8} value={rank} onChange={(e) => setRank(Number(e.target.value))} />
                </label>
                <label className="field">
                  <span>Learning rate</span>
                  <input
                    type="number"
                    step={0.0001}
                    value={learningRate}
                    onChange={(e) => setLearningRate(Number(e.target.value))}
                  />
                </label>
                <label className="field">
                  <span>Seed</span>
                  <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
                </label>
                <label className="field">
                  <span>Validation every (steps)</span>
                  <input
                    type="number"
                    min={50}
                    step={50}
                    value={validationInterval}
                    onChange={(e) => setValidationInterval(Number(e.target.value))}
                  />
                </label>
                <label className="field">
                  <span>Checkpoint every (steps)</span>
                  <input
                    type="number"
                    min={50}
                    step={50}
                    value={checkpointInterval}
                    onChange={(e) => setCheckpointInterval(Number(e.target.value))}
                  />
                </label>
                <label className="checkbox-row span-2">
                  <input type="checkbox" checked={lowRam} onChange={(e) => setLowRam(e.target.checked)} />
                  Low RAM mode (gradient checkpointing)
                </label>
                <label className="field span-2">
                  <span>Validation prompts (one per line)</span>
                  <textarea
                    rows={3}
                    value={validationPrompts}
                    onChange={(e) => setValidationPrompts(e.target.value)}
                  />
                </label>
              </div>

              <div className="train-summary">
                <div>
                  <strong>{name}</strong> · {selectedPreset?.label || preset} · {steps} steps · rank {rank}
                </div>
                <div className="train-summary-meta">
                  {videoCount} target{isV2v ? "" : ""} video{videoCount !== 1 ? "s" : ""}
                  {isV2v ? ` · ${referenceCount} reference${referenceCount !== 1 ? "s" : ""}` : ""}
                  {" · "}
                  {width}×{height} · {maxFrames} frames
                </div>
              </div>

              <div className="train-actions">
                <button type="button" className="btn-secondary" onClick={() => setStep("preprocess")}>
                  Back
                </button>
                <button
                  type="button"
                  className="btn-primary"
                  disabled={submitting || !health?.trainer_installed}
                  onClick={startTraining}
                >
                  {submitting ? "Starting…" : "Start training"}
                </button>
              </div>
            </section>
          )}

          {step === "runs" && (
            <section className="train-section">
              <h2>Runs</h2>
              {!activeJob && jobs.length === 0 && (
                <p className="train-section-lead">No training jobs yet. Configure a run and start from the Train step.</p>
              )}

              {jobs.length > 0 && (
                <div className="job-list">
                  {jobs.map((j) => (
                    <button
                      key={j.id}
                      type="button"
                      className={`job-card ${activeJobId === j.id ? "active" : ""}`}
                      onClick={() => setActiveJobId(j.id)}
                    >
                      <span className="job-card-name">{j.name}</span>
                      <span className={`job-status status-${j.status}`}>{j.status}</span>
                      <span className="job-card-meta">{new Date(j.created_at).toLocaleString()}</span>
                    </button>
                  ))}
                </div>
              )}

              {activeJob && (
                <div className="job-detail">
                  <div className="job-detail-header">
                    <div>
                      <h3>{activeJob.name}</h3>
                      <p className="job-phase">
                        {phaseLabel(activeJob.phase)} · {activeJob.preset}
                      </p>
                    </div>
                    <div className="artifact-actions">
                      {["interrupted", "failed"].includes(activeJob.status) && (
                        <button
                          type="button"
                          className="btn-primary"
                          disabled={resuming || !!health?.generation_active}
                          onClick={() => handleResume()}
                        >
                          {resuming ? "Resuming…" : "Resume"}
                        </button>
                      )}
                      {["queued", "running"].includes(activeJob.status) && (
                        <button type="button" className="btn-danger" onClick={handleCancel}>
                          Cancel
                        </button>
                      )}
                    </div>
                  </div>

                  {["queued", "running", "slicing", "preprocessing", "training", "starting"].includes(
                    activeJob.phase || "",
                  ) &&
                    activeJob.status !== "interrupted" && (
                    <div className="progress-block">
                      <div className="progress-bar">
                        <div
                          className="progress-fill"
                          style={{
                            width: activeJob.phase === "training" ? `${trainProgress}%` : "12%",
                          }}
                        />
                      </div>
                      {activeJob.phase === "training" && (
                        <div className="progress-stats">
                          <span>
                            Step {activeJob.step || 0} / {activeJob.total_steps || steps}
                          </span>
                          {activeJob.loss != null && <span>Loss {activeJob.loss.toFixed(4)}</span>}
                          <span>ETA {formatEta(activeJob.eta_s)}</span>
                        </div>
                      )}
                    </div>
                  )}

                  {activeJob.status === "done" && activeJob.artifact_url && (
                    <div className="artifact-card">
                      <div>
                        <strong>LoRA ready</strong>
                        <div className="artifact-name">{activeJob.artifact_name}</div>
                      </div>
                      <div className="artifact-actions">
                        <a className="btn-secondary" href={activeJob.artifact_url} download>
                          Download
                        </a>
                        <button
                          type="button"
                          className="btn-primary"
                          disabled={registering || !!activeJob.registered_lora_id}
                          onClick={handleRegister}
                        >
                          {activeJob.registered_lora_id ? "Added to library" : registering ? "Adding…" : "Use in Generate"}
                        </button>
                      </div>
                    </div>
                  )}

                  {activeJob.error && <div className="train-error">{activeJob.error}</div>}

                  {(activeJob.validation_clips?.length ?? 0) > 0 && (
                    <div className="validation-gallery">
                      <h4>Validation previews</h4>
                      <div className="validation-grid">
                        {activeJob.validation_clips!.map((v, i) => (
                          <div key={`${v.url}-${i}`} className="validation-card">
                            <video src={v.url} controls playsInline loop muted />
                            <span className="validation-step">Step {v.step}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              <div className="train-actions">
                <button type="button" className="btn-secondary" onClick={() => setStep("train")}>
                  New run setup
                </button>
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
