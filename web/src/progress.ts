import type { ModelProgress, ProgressState } from "./types";

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
    elapsed_s,
  };
}

export function formatProgressMessage(
  mp?: ModelProgress | null,
  elapsed_s?: number,
): string {
  if (!mp?.stage && !mp?.step) {
    return elapsed_s != null ? `Generating… ${elapsed_s}s` : "Generating…";
  }
  const parts: string[] = [];
  const stage = mp?.stage || "generating";
  parts.push(stage.charAt(0).toUpperCase() + stage.slice(1));
  if (mp?.step != null && mp?.total != null) {
    parts.push(`step ${mp.step}/${mp.total}`);
  }
  if (mp?.pct != null) {
    parts.push(`${mp.pct}%`);
  }
  if (mp?.eta_s != null) {
    parts.push(`~${mp.eta_s}s left`);
  }
  if (mp?.avg_step_s != null && mp.step != null && mp.total != null) {
    parts.push(`${mp.avg_step_s}s/step`);
  }
  if (elapsed_s != null) {
    parts.push(`${elapsed_s}s elapsed`);
  }
  return parts.join(" · ");
}

export function applyProgressEvent(
  prev: ProgressState | null,
  msg: Record<string, unknown>,
): ProgressState {
  const mp = (msg.model_progress as ModelProgress | undefined) ?? undefined;
  const elapsed_s =
    typeof msg.elapsed_s === "number" ? msg.elapsed_s : prev?.elapsed_s;
  const phase =
    typeof msg.phase === "string"
      ? msg.phase
      : mp?.stage ?? prev?.phase ?? "generating";
  return {
    phase,
    message: formatProgressMessage(mp, elapsed_s),
    ...progressFromModel(mp, elapsed_s),
  };
}
