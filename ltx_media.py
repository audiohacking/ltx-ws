"""
PyAV-backed audio/video helpers for ltx-ws.

All media trimming, segmentation, concat, muxing, and inference audio loading
goes through this module (PyAV / ``pip install av``).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger(__name__)

try:
    import av
    from av.audio.resampler import AudioResampler

    _AV_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via media_available() in tests
    av = None  # type: ignore[assignment,misc]
    AudioResampler = None  # type: ignore[assignment,misc]
    _AV_AVAILABLE = False

AUDIO_OUTPUT_RATE = 44100
AUDIO_OUTPUT_LAYOUT = "stereo"
AUDIO_OUTPUT_FORMAT = "s16"
_MIN_WAV_BYTES = 44


def media_available() -> bool:
    """True when PyAV is importable."""
    return _AV_AVAILABLE


def require_media() -> None:
    if not _AV_AVAILABLE:
        raise RuntimeError("PyAV is required — install with: pip install av")


def probe_audio_duration(path: Path | str) -> float | None:
    """Return audio duration in seconds, or None when probing fails."""
    require_media()
    path = Path(path)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
            if duration > 0:
                return duration
        if container.duration:
            duration = float(container.duration) / float(av.time_base)
            if duration > 0:
                return duration
        last_time = 0.0
        for frame in container.decode(stream):
            if frame.time is not None:
                last_time = max(last_time, float(frame.time))
            elif frame.pts is not None and frame.time_base is not None:
                last_time = max(last_time, float(frame.pts * frame.time_base))
        return last_time if last_time > 0 else None


def _decode_audio_planar_f32(
    path: Path | str,
    *,
    target_sample_rate: int,
    mono: bool,
) -> tuple[Any, int] | None:
    """Decode all audio from ``path`` to planar float32 (channels, samples)."""
    import numpy as np

    require_media()
    layout = "mono" if mono else "stereo"
    resampler = AudioResampler(format="fltp", layout=layout, rate=target_sample_rate)
    parts: list[Any] = []

    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                parts.append(np.asarray(arr, dtype=np.float32))
        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray()
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            parts.append(np.asarray(arr, dtype=np.float32))

    if not parts:
        return None
    data = np.concatenate(parts, axis=1)
    if data.shape[1] == 0:
        return None
    return data, target_sample_rate


def _write_pcm_wav_from_planar_f32(
    dst: Path | str,
    data: Any,
    *,
    sample_rate: int,
) -> None:
    """Encode planar float32 (channels, samples) to PCM WAV."""
    import numpy as np

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    resampler = AudioResampler(
        format=AUDIO_OUTPUT_FORMAT,
        layout=AUDIO_OUTPUT_LAYOUT,
        rate=sample_rate,
    )
    chunk = 4096
    n_samples = int(data.shape[1])

    with av.open(str(dst), "w", format="wav") as out_container:
        out_stream = out_container.add_stream(
            "pcm_s16le",
            rate=sample_rate,
            layout=AUDIO_OUTPUT_LAYOUT,
        )
        for offset in range(0, n_samples, chunk):
            block = np.asarray(data[:, offset : offset + chunk], dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(
                block,
                format="fltp",
                layout=AUDIO_OUTPUT_LAYOUT,
            )
            frame.sample_rate = sample_rate
            for resampled in resampler.resample(frame):
                for packet in out_stream.encode(resampled):
                    out_container.mux(packet)
        for resampled in resampler.resample(None):
            for packet in out_stream.encode(resampled):
                out_container.mux(packet)
        for packet in out_stream.encode(None):
            out_container.mux(packet)


def trim_audio_start(
    src: Path | str,
    dst: Path | str,
    start_seconds: float,
) -> Path:
    """Write ``src`` from ``start_seconds`` onward to PCM WAV at ``dst``."""
    require_media()
    src = Path(src)
    dst = Path(dst)
    start_seconds = max(0.0, float(start_seconds))

    decoded = _decode_audio_planar_f32(
        src,
        target_sample_rate=AUDIO_OUTPUT_RATE,
        mono=False,
    )
    if decoded is None:
        raise RuntimeError(f"No audio stream found in {src}")

    data, sample_rate = decoded
    start_sample = int(start_seconds * sample_rate)
    if start_sample >= data.shape[1]:
        raise RuntimeError(f"Audio trim produced empty output: {dst}")
    _write_pcm_wav_from_planar_f32(dst, data[:, start_sample:], sample_rate=sample_rate)

    if not dst.is_file() or dst.stat().st_size <= _MIN_WAV_BYTES:
        raise RuntimeError(f"Audio trim produced empty output: {dst}")
    return dst


def split_audio(
    src: Path | str,
    out_dir: Path | str,
    *,
    segment_seconds: float,
    required_segments: int,
    suffix: str = ".wav",
) -> list[Path]:
    """Split ``src`` into fixed-duration PCM WAV segments."""
    require_media()
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segment_seconds = max(0.25, float(segment_seconds))
    if required_segments < 1:
        raise ValueError("required_segments must be >= 1")

    resampler = AudioResampler(
        format=AUDIO_OUTPUT_FORMAT,
        layout=AUDIO_OUTPUT_LAYOUT,
        rate=AUDIO_OUTPUT_RATE,
    )

    segments: list[Path] = []
    seg_index = 0
    seg_elapsed = 0.0
    out_container = None
    out_stream = None

    def _open_segment() -> None:
        nonlocal out_container, out_stream, seg_index, seg_elapsed
        out_path = out_dir / f"seg_{seg_index:04d}{suffix}"
        segments.append(out_path)
        out_container = av.open(str(out_path), "w", format="wav")
        out_stream = out_container.add_stream(
            "pcm_s16le",
            rate=AUDIO_OUTPUT_RATE,
            layout=AUDIO_OUTPUT_LAYOUT,
        )
        seg_elapsed = 0.0

    def _close_segment() -> None:
        nonlocal out_container, out_stream
        if out_container is None or out_stream is None:
            return
        for packet in out_stream.encode(None):
            out_container.mux(packet)
        out_container.close()
        out_container = None
        out_stream = None

    _open_segment()

    with av.open(str(src)) as in_container:
        if not in_container.streams.audio:
            _close_segment()
            raise RuntimeError(f"No audio stream found in {src}")
        in_stream = in_container.streams.audio[0]
        for frame in in_container.decode(in_stream):
            frame_seconds = 0.0
            if frame.samples and frame.sample_rate:
                frame_seconds = float(frame.samples) / float(frame.sample_rate)
            elif frame.time is not None and seg_elapsed == 0.0:
                frame_seconds = 0.1

            if seg_elapsed + frame_seconds > segment_seconds and seg_elapsed > 0:
                _close_segment()
                seg_index += 1
                _open_segment()

            for resampled in resampler.resample(frame):
                for packet in out_stream.encode(resampled):
                    out_container.mux(packet)
            seg_elapsed += frame_seconds

    _close_segment()

    if len(segments) < required_segments:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise RuntimeError(
            f"Audio produced {len(segments)} segment(s), but {required_segments} "
            "clip(s) are queued. Increase source audio length or shorten clips."
        )
    return segments


def _stream_duration_packets(stream: av.stream.Stream, last_packet: av.packet.Packet) -> int:
    if stream.duration:
        return int(stream.duration)
    if last_packet.dts is not None:
        return int(last_packet.dts) + 1
    if last_packet.pts is not None:
        return int(last_packet.pts) + 1
    return 0


def concat_videos(
    inputs: Sequence[Path | str],
    output: Path | str,
    *,
    reencode_h265: bool = False,
) -> Path:
    """Concatenate MP4/MOV inputs. Stream-copy by default; optional libx265 reencode."""
    require_media()
    paths = [Path(p) for p in inputs]
    if len(paths) < 2:
        raise ValueError("concat_videos requires at least two inputs")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if reencode_h265:
        return _concat_videos_reencode_h265(paths, output)
    return _concat_videos_copy(paths, output)


def _concat_videos_copy(paths: list[Path], output: Path) -> Path:
    with av.open(str(output), "w") as out_container:
        out_video: av.stream.Stream | None = None
        out_audio: av.stream.Stream | None = None
        video_offset = 0
        audio_offset = 0

        for path in paths:
            with av.open(str(path)) as in_container:
                if not in_container.streams.video:
                    raise RuntimeError(f"No video stream in {path}")
                in_video = in_container.streams.video[0]
                in_audio = in_container.streams.audio[0] if in_container.streams.audio else None

                if out_video is None:
                    out_video = out_container.add_stream(template=in_video)
                    if in_audio is not None:
                        out_audio = out_container.add_stream(template=in_audio)
                elif in_audio is not None and out_audio is None:
                    out_audio = out_container.add_stream(template=in_audio)

                last_v: av.packet.Packet | None = None
                for packet in in_container.demux(in_video):
                    if packet.dts is None and packet.pts is None:
                        continue
                    if packet.dts is not None:
                        packet.dts += video_offset
                    if packet.pts is not None:
                        packet.pts += video_offset
                    packet.stream = out_video
                    out_container.mux(packet)
                    last_v = packet

                if last_v is not None:
                    video_offset += _stream_duration_packets(in_video, last_v)

                if in_audio is not None and out_audio is not None:
                    last_a: av.packet.Packet | None = None
                    for packet in in_container.demux(in_audio):
                        if packet.dts is None and packet.pts is None:
                            continue
                        if packet.dts is not None:
                            packet.dts += audio_offset
                        if packet.pts is not None:
                            packet.pts += audio_offset
                        packet.stream = out_audio
                        out_container.mux(packet)
                        last_a = packet
                    if last_a is not None:
                        audio_offset += _stream_duration_packets(in_audio, last_a)

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Video concat produced empty output: {output}")
    return output


def _concat_videos_reencode_h265(paths: list[Path], output: Path) -> Path:
    with av.open(str(paths[0])) as first:
        in_v = first.streams.video[0]
        rate = in_v.average_rate or in_v.codec_context.framerate or 24
        width = in_v.codec_context.width
        height = in_v.codec_context.height

    with av.open(str(output), "w") as out_container:
        out_v = out_container.add_stream("libx265", rate=rate, width=width, height=height)
        out_v.options = {"crf": "28", "preset": "faster"}
        for path in paths:
            with av.open(str(path)) as in_container:
                in_v = in_container.streams.video[0]
                for frame in in_container.decode(in_v):
                    frame.pts = None
                    for packet in out_v.encode(frame):
                        out_container.mux(packet)
        for packet in out_v.encode(None):
            out_container.mux(packet)

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Video reencode concat produced empty output: {output}")
    return output


def mux_audio_into_video(
    video_path: Path | str,
    audio_path: Path | str,
    output_path: Path | str,
    *,
    duration_s: float,
) -> Path:
    """Mux ``audio_path`` onto a (usually silent) ``video_path``; copy video, encode AAC audio."""
    require_media()
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_s = max(0.1, float(duration_s))

    resampler = AudioResampler(
        format="fltp",
        layout=AUDIO_OUTPUT_LAYOUT,
        rate=AUDIO_OUTPUT_RATE,
    )

    with av.open(str(video_path)) as vin, av.open(str(audio_path)) as ain:
        if not vin.streams.video:
            raise RuntimeError(f"No video stream in {video_path}")
        if not ain.streams.audio:
            raise RuntimeError(f"No audio stream in {audio_path}")
        v_in = vin.streams.video[0]
        a_in = ain.streams.audio[0]

        with av.open(str(output_path), "w") as out:
            v_out = out.add_stream(template=v_in)
            a_out = out.add_stream("aac", rate=AUDIO_OUTPUT_RATE, layout=AUDIO_OUTPUT_LAYOUT)

            for packet in vin.demux(v_in):
                if packet.dts is None and packet.pts is None:
                    continue
                if packet.pts is not None and packet.time is not None and packet.time > duration_s:
                    break
                packet.stream = v_out
                out.mux(packet)

            for frame in ain.decode(a_in):
                if frame.time is not None and frame.time > duration_s:
                    break
                for resampled in resampler.resample(frame):
                    for packet in a_out.encode(resampled):
                        out.mux(packet)
            for packet in a_out.encode(None):
                out.mux(packet)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Audio mux produced empty output: {output_path}")
    return output_path


def trim_audio_to_temp(audio_path: str, start_seconds: float) -> tuple[Path, Path]:
    """Trim ``audio_path`` into a scratch WAV; returns (file, temp_dir)."""
    from ltx_paths import mk_scratch_dir

    temp_dir = mk_scratch_dir("ltx_audio_trim_")
    out_path = temp_dir / "segment.wav"
    trim_audio_start(audio_path, out_path, start_seconds)
    return out_path, temp_dir


def load_audio_for_inference(
    path: str | Path,
    target_sample_rate: int = 16000,
    start_time: float = 0.0,
    max_duration: float | None = None,
    mono: bool = False,
) -> Any | None:
    """
    Load audio for MLX inference via PyAV.

    Same contract as ``ltx_core_mlx.utils.audio.load_audio``; ltx-ws patches
    upstream to call this implementation exclusively (no system ffmpeg).
    """
    import mlx.core as mx

    try:
        from ltx_core_mlx.utils.audio import AudioData
    except ImportError:  # pragma: no cover - only when ltx-core-mlx missing
        return None

    decoded = _decode_audio_planar_f32(
        path,
        target_sample_rate=target_sample_rate,
        mono=mono,
    )
    if decoded is None:
        return None

    data, sample_rate = decoded
    start_sample = int(max(0.0, float(start_time)) * sample_rate)
    if start_sample >= data.shape[1]:
        return None
    data = data[:, start_sample:]
    if max_duration is not None:
        end_sample = int(float(max_duration) * sample_rate)
        data = data[:, : max(0, end_sample)]
    if data.shape[1] == 0:
        return None

    waveform = mx.array(data)[None, :, :]
    return AudioData(waveform=waveform, sample_rate=sample_rate)
