import { useEffect, useState } from "react";
import type { SystemStatus } from "./types";

const API = "";

async function fetchSystemStatus(): Promise<SystemStatus> {
  const r = await fetch(`${API}/api/system/status`);
  if (!r.ok) throw new Error("Failed to load system status");
  return r.json();
}
function phaseLabel(phase: string): string {
  switch (phase) {
    case "downloading_model":
      return "Downloading model";
    case "downloading_lora":
      return "Downloading LoRA";
    case "loading_mlx":
      return "Loading MLX";
    case "loading_pipeline":
      return "Loading pipeline";
    case "resolving_loras":
      return "Resolving LoRAs";
    case "ready":
      return "Ready";
    case "error":
      return "Error";
    default:
      return phase;
  }
}

export default function SystemStatusBar() {
  const [status, setStatus] = useState<SystemStatus | null>(null);

  useEffect(() => {
    let closed = false;
    fetchSystemStatus().then((s) => {
      if (!closed) setStatus(s);
    });
    const es = new EventSource(`${API}/api/system/events`);
    es.onmessage = (ev) => {
      try {
        setStatus(JSON.parse(ev.data) as SystemStatus);
      } catch {
        /* ignore */
      }
    };
    es.onerror = () => es.close();
    return () => {
      closed = true;
      es.close();
    };
  }, []);

  if (!status) {
    return null;
  }

  if (status.phase === "idle") {
    if (!status.frozen) return null;
    return (
      <div className="system-status busy" title={status.message}>
        <span className="system-dot busy" />
        <span className="system-text">{status.message || "Starting…"}</span>
      </div>
    );
  }

  if (status.phase === "ready") {
    if (status.model) {
      return (
        <div className="system-status ready" title={status.detail || status.model}>
          <span className="system-dot ok" />
          {status.model}
        </div>
      );
    }
    return null;
  }

  const pct =
    status.pct != null && status.pct > 0 ? `${Math.round(status.pct)}%` : null;

  return (
    <div
      className={`system-status ${status.phase === "error" ? "error" : "busy"}`}
      title={status.detail || status.message}
    >
      <span className={`system-dot ${status.phase === "error" ? "err" : "busy"}`} />
      <span className="system-text">
        {phaseLabel(status.phase)}
        {pct ? ` ${pct}` : ""}
        {status.pipeline ? ` · ${status.pipeline}` : ""}
      </span>
    </div>
  );
}
