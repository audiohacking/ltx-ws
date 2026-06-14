import type { ModelProgress, ProgressState } from "./types";

/** tqdm-style clock: 54 → "00:54", 125 → "02:05" */
export function formatMmSs(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

export function progressFromModel(
  mp?: ModelProgress | null,
  elapsed_s?: number,
): Partial<ProgressState> {
  if (!mp) {
    return elapsed_s != null ? { elapsed_s } : {};
  }
  return {
    stage: mp.stage,
    step: mp.step,
    total: mp.total,
    pct: mp.pct,
    eta_s: mp.eta_s,
    elapsed_s: mp.elapsed_s ?? elapsed_s,
  };
}

/** Match CLI tqdm: ``2/8 [00:18<00:54, 9.11s/it]`` — lead with remaining time. */
export function formatProgressMessage(
  mp?: ModelProgress | null,
  elapsed_s?: number,
): string {
  if (!mp?.stage && mp?.step == null) {
    if (elapsed_s != null) {
      return `Generating… ${formatMmSs(elapsed_s)} elapsed`;
    }
    return "Generating…";
  }
  const parts: string[] = [];
  const stage = mp?.stage || "generating";
  parts.push(stage.charAt(0).toUpperCase() + stage.slice(1));

  if (mp?.step != null && mp?.total != null) {
    parts.push(`${mp.step}/${mp.total}`);
  }

  if (mp?.eta_s != null) {
    parts.push(`${formatMmSs(mp.eta_s)} remaining`);
  } else if (mp?.pct != null) {
    parts.push(`${mp.pct}%`);
  }

  if (mp?.avg_step_s != null) {
    parts.push(`${mp.avg_step_s}s/it`);
  }

  if (elapsed_s != null && mp?.eta_s == null) {
    parts.push(`${formatMmSs(elapsed_s)} elapsed`);
  }

  return parts.join(" · ");
}

export function applyProgressEvent(
  prev: ProgressState | null,
  msg: Record<string, unknown>,
): ProgressState {
  const mp = (msg.model_progress as ModelProgress | undefined) ?? undefined;
  const wall_elapsed =
    typeof msg.elapsed_s === "number" ? msg.elapsed_s : prev?.elapsed_s;
  const phase =
    typeof msg.phase === "string"
      ? msg.phase
      : mp?.stage ?? prev?.phase ?? "generating";
  return {
    phase,
    message: formatProgressMessage(mp, wall_elapsed),
    ...progressFromModel(mp, wall_elapsed),
  };
}
