import type { Clip, Config } from "./types";

/** Strip merged suffix from stored prompt for editor display. */
export function clipDisplayPrompt(prompt: string): string {
  return prompt.replace(/\s*\(×\d+ merged\)\s*$/i, "").trim();
}

export function resolutionIdForClip(
  clip: Clip,
  config: Config | null,
): string {
  if (!clip.width || !clip.height) return "704x480";
  const match = config?.resolution_presets.find(
    (r) => r.width === clip.width && r.height === clip.height,
  );
  return match?.id ?? `${clip.width}x${clip.height}`;
}

export function durationIdForClip(clip: Clip, config: Config | null): string {
  if (clip.duration_seconds != null) {
    const match = config?.duration_presets.find(
      (d) => d.seconds === clip.duration_seconds,
    );
    if (match) return match.id;
    return `${clip.duration_seconds}s`;
  }
  if (clip.num_frames && config?.defaults.fps) {
    const seconds = clip.num_frames / config.defaults.fps;
    const match = config?.duration_presets.find((d) => d.seconds === seconds);
    if (match) return match.id;
  }
  return "5s";
}

export interface ClipEditorSnapshot {
  prompt: string;
  mode: string;
  resolutionId: string;
  durationId: string;
  clipMultiplier: number;
  numSteps: number;
  seed: string;
  autocontinue: boolean;
  autoconcat: boolean;
}

export function snapshotFromClip(
  clip: Clip,
  config: Config | null,
  defaults: { numSteps: number },
): ClipEditorSnapshot {
  return {
    prompt: clipDisplayPrompt(clip.prompt),
    mode: clip.mode || "generate",
    resolutionId: resolutionIdForClip(clip, config),
    durationId: durationIdForClip(clip, config),
    clipMultiplier: clip.clip_count ?? 1,
    numSteps: clip.num_steps ?? defaults.numSteps,
    seed: clip.seed != null ? String(clip.seed) : "",
    autocontinue: clip.autocontinue ?? (clip.clip_count ?? 1) > 1,
    autoconcat: clip.autoconcat ?? (clip.clip_count ?? 1) > 1,
  };
}
