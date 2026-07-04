import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Props = {
  fileName: string | null;
  previewUrl: string | null;
  durationSeconds: number | null;
  startSeconds: number;
  clipDurationSeconds: number;
  maxStart: number;
  trimAvailable: boolean;
  disabled?: boolean;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  onFileSelected: (file: File) => void;
  onStartChange: (seconds: number) => void;
};

export function AudioTrimControl({
  fileName,
  previewUrl,
  durationSeconds,
  startSeconds,
  clipDurationSeconds,
  maxStart,
  trimAvailable,
  disabled = false,
  fileInputRef,
  onFileSelected,
  onStartChange,
}: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const segmentEnd = startSeconds + clipDurationSeconds;

  const timeline = useMemo(() => {
    const total =
      durationSeconds && durationSeconds > 0
        ? durationSeconds
        : Math.max(maxStart + clipDurationSeconds, 1);
    const startPct = Math.min(100, Math.max(0, (startSeconds / total) * 100));
    const widthPct = Math.min(100 - startPct, (clipDurationSeconds / total) * 100);
    return { total, startPct, widthPct };
  }, [clipDurationSeconds, durationSeconds, maxStart, startSeconds]);

  const stopPreview = useCallback(() => {
    const el = audioRef.current;
    if (!el) return;
    el.pause();
    el.currentTime = startSeconds;
    setIsPlaying(false);
  }, [startSeconds]);

  const playSegmentPreview = useCallback(() => {
    const el = audioRef.current;
    if (!el || !previewUrl) return;
    if (isPlaying) {
      stopPreview();
      return;
    }
    el.currentTime = startSeconds;
    void el.play().then(() => setIsPlaying(true)).catch(() => setIsPlaying(false));
  }, [isPlaying, previewUrl, startSeconds, stopPreview]);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;

    const onTimeUpdate = () => {
      if (el.currentTime >= segmentEnd - 0.05) {
        el.pause();
        el.currentTime = Math.min(segmentEnd, el.duration || segmentEnd);
        setIsPlaying(false);
      }
    };
    const onPause = () => setIsPlaying(false);
    const onEnded = () => setIsPlaying(false);

    el.addEventListener("timeupdate", onTimeUpdate);
    el.addEventListener("pause", onPause);
    el.addEventListener("ended", onEnded);
    return () => {
      el.removeEventListener("timeupdate", onTimeUpdate);
      el.removeEventListener("pause", onPause);
      el.removeEventListener("ended", onEnded);
    };
  }, [previewUrl, segmentEnd]);

  useEffect(() => {
    stopPreview();
  }, [previewUrl, startSeconds, stopPreview]);

  return (
    <div className="audio-trim-panel">
      <label className="media-upload">
        <span className="media-upload-label">Source audio (required)</span>
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          disabled={disabled}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFileSelected(f);
          }}
        />
        <span className="media-upload-hint">{fileName ?? "Choose audio file…"}</span>
      </label>

      {previewUrl && (
        <>
          <audio ref={audioRef} className="audio-trim-audio-hidden" preload="metadata" src={previewUrl} />

          <div className="audio-start-control">
            <div className="audio-start-header">
              <span className="audio-start-label">Audio start</span>
              <span className="audio-start-value">
                {startSeconds.toFixed(1)}s → {segmentEnd.toFixed(1)}s
                {durationSeconds ? ` · ${Math.floor(durationSeconds)}s total` : ""}
              </span>
              <button
                type="button"
                className={`btn-secondary audio-trim-preview-btn${isPlaying ? " is-playing" : ""}`}
                disabled={disabled}
                aria-pressed={isPlaying}
                onClick={playSegmentPreview}
              >
                {isPlaying ? "Stop" : "Preview"}
              </button>
            </div>

            <div className="audio-trim-timeline" aria-hidden={!durationSeconds}>
              <div className="audio-trim-timeline-track">
                <div
                  className="audio-trim-timeline-segment"
                  style={{
                    left: `${timeline.startPct}%`,
                    width: `${Math.max(timeline.widthPct, 0.5)}%`,
                  }}
                />
              </div>
            </div>

            <input
              type="range"
              className="audio-start-slider"
              min={0}
              max={maxStart}
              step={0.1}
              value={Math.min(startSeconds, maxStart)}
              disabled={disabled}
              aria-label="Audio start offset in seconds"
              onChange={(e) => onStartChange(Number(e.target.value))}
            />
            <div className="audio-start-input-row">
              <input
                type="number"
                min={0}
                max={maxStart}
                step={0.1}
                value={startSeconds}
                disabled={disabled}
                aria-label="Audio start seconds"
                onChange={(e) =>
                  onStartChange(Math.min(maxStart, Math.max(0, Number(e.target.value) || 0)))
                }
              />
              <span className="audio-start-hint">
                {clipDurationSeconds.toFixed(1)}s clip · local preview only until generate
              </span>
            </div>
            {startSeconds > 0 && !trimAvailable && (
              <p className="hint hint-inline">
                Audio start requires PyAV on the server (pip install av).
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
