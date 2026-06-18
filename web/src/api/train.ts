import type { TrainHealth, TrainJob, TrainPreset } from "../types";

export async function fetchTrainHealth(): Promise<TrainHealth> {
  const res = await fetch("/api/train/health");
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

export async function fetchTrainPresets(): Promise<TrainPreset[]> {
  const res = await fetch("/api/train/presets");
  if (!res.ok) throw new Error(`Presets failed (${res.status})`);
  const data = await res.json();
  return data.presets ?? [];
}

export async function fetchTrainJobs(): Promise<TrainJob[]> {
  const res = await fetch("/api/train/jobs");
  if (!res.ok) throw new Error(`Jobs list failed (${res.status})`);
  const data = await res.json();
  return data.jobs ?? [];
}

export interface TrainManifest {
  name: string;
  preset: string;
  model_id: string;
  slice: {
    enabled: boolean;
    interval: number;
    res: string;
    fps: number;
    fit: string;
    caption_template?: string;
    max_clips?: number;
  };
  preprocess: {
    width: number;
    height: number;
    max_frames: number;
    with_audio: boolean;
    frame_rate: number;
    reference_downscale_factor?: number;
  };
  train: {
    steps: number;
    rank: number;
    learning_rate: number;
    validation_prompts: string[];
    validation_interval: number;
    checkpoint_interval: number;
    low_ram: boolean;
    seed: number;
  };
}

export async function createTrainJob(
  manifest: TrainManifest,
  targetFiles: File[],
  referenceFiles: File[] = [],
): Promise<{ job_id: string; name: string; preset: string }> {
  const form = new FormData();
  form.append("manifest", JSON.stringify(manifest));
  for (const file of targetFiles) {
    form.append("videos", file, file.name);
  }
  for (const file of referenceFiles) {
    form.append("references", file, file.name);
  }
  const res = await fetch("/api/train/jobs", { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Create job failed (${res.status})`);
  }
  return res.json();
}

export async function resumeTrainJob(jobId: string): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`/api/train/jobs/${jobId}/resume`, { method: "POST" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Resume failed (${res.status})`);
  }
  return res.json();
}

export async function cancelTrainJob(jobId: string): Promise<void> {
  const res = await fetch(`/api/train/jobs/${jobId}/cancel`, { method: "POST" });
  if (!res.ok) throw new Error(`Cancel failed (${res.status})`);
}

export async function registerTrainedLora(
  jobId: string,
  label: string,
  scale = 1.0,
): Promise<{ id: string; spec: string }> {
  const res = await fetch(`/api/train/jobs/${jobId}/register-lora`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label, scale }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Register failed (${res.status})`);
  }
  return res.json();
}

export type TrainEventHandler = (event: Record<string, unknown>) => void;

export function subscribeTrainJob(jobId: string, onEvent: TrainEventHandler): () => void {
  const es = new EventSource(`/api/train/jobs/${jobId}/events`);

  es.onmessage = (msg) => {
    try {
      const event = JSON.parse(msg.data) as Record<string, unknown>;
      onEvent(event);
    } catch {
      /* ignore malformed */
    }
  };

  es.onerror = () => {
    es.close();
  };

  return () => es.close();
}
