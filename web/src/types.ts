export interface ModelOption {
  id: string;
  label: string;
  repo: string;
}

export interface PresetOption {
  id: string;
  label: string;
  width?: number;
  height?: number;
  seconds?: number;
}

export interface Config {
  server_connected: boolean;
  server_url: string;
  preferred_model: string;
  models: ModelOption[];
  resolution_presets: PresetOption[];
  duration_presets: PresetOption[];
  generation_modes: { id: string; label: string }[];
  defaults: {
    num_frames: number;
    width: number;
    height: number;
    num_steps: number;
    fps: number;
  };
  model_note: string;
  clip_multiplier_max?: number;
  embedded?: boolean;
  web_url?: string;
  active_model?: string;
}

export interface Clip {
  id: string;
  prompt: string;
  label: string;
  video_url: string;
  filename: string;
  chain_id: string;
  clip_index: number;
  mode: string;
  status: string;
  created_at: string;
  elapsed_s?: number;
  bytes?: number;
  error?: string;
  num_frames?: number;
  width?: number;
  height?: number;
  seed?: number;
}

export interface ProgressState {
  phase: string;
  message: string;
  elapsed_s?: number;
  kb?: number;
}
