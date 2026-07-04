"""
PyAV-backed audio/video helpers for ltx-ws.

All media trimming, segmentation, concat, and muxing goes through this module
instead of invoking the ffmpeg/sox CLI.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

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


def _frame_bounds(frame: av.AudioFrame, decoded_seconds: float) -> tuple[float, float]:
    frame_start = decoded_seconds
    if frame.time is not None:
        frame_start = float(frame.time)
    if frame.samples and frame.sample_rate:
        frame_len = float(frame.samples) / float(frame.sample_rate)
    else:
        frame_len = 0.0
    return frame_start, frame_start + frame_len


def _slice_audio_frame(frame: av.AudioFrame, skip_samples: int) -> av.AudioFrame:
    if skip_samples <= 0:
        return frame
    if skip_samples >= frame.samples:
        raise ValueError("skip_samples exceeds frame length")
    sliced = frame.to_ndarray()[:, skip_samples:]
    out = av.AudioFrame.from_ndarray(sliced, format=frame.format, layout=frame.layout)
    out.sample_rate = frame.sample_rate
    if frame.pts is not None:
        out.pts = frame.pts + skip_samples
    if frame.time_base is not None:
        out.time_base = frame.time_base
    return out


def trim_audio_start(
    src: Path | str,
    dst: Path | str,
    start_seconds: float,
) -> Path:
    """Write ``src`` from ``start_seconds`` onward to PCM WAV at ``dst``."""
    require_media()
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    start_seconds = max(0.0, float(start_seconds))

    resampler = AudioResampler(
        format=AUDIO_OUTPUT_FORMAT,
        layout=AUDIO_OUTPUT_LAYOUT,
        rate=AUDIO_OUTPUT_RATE,
    )

    with av.open(str(src)) as in_container:
        if not in_container.streams.audio:
            raise RuntimeError(f"No audio stream found in {src}")
        in_stream = in_container.streams.audio[0]
        decoded_seconds = 0.0

        with av.open(str(dst), "w", format="wav") as out_container:
            out_stream = out_container.add_stream(
                "pcm_s16le",
                rate=AUDIO_OUTPUT_RATE,
                layout=AUDIO_OUTPUT_LAYOUT,
            )
            for frame in in_container.decode(in_stream):
                frame_start, frame_end = _frame_bounds(frame, decoded_seconds)
                if frame.samples and frame.sample_rate:
                    decoded_seconds = max(
                        decoded_seconds,
                        frame_end,
                    )

                if frame_end <= start_seconds:
                    continue

                if frame_start < start_seconds:
                    skip_samples = min(
                        frame.samples,
                        int((start_seconds - frame_start) * frame.sample_rate),
                    )
                    if skip_samples >= frame.samples:
                        continue
                    frame = _slice_audio_frame(frame, skip_samples)

                for resampled in resampler.resample(frame):
                    for packet in out_stream.encode(resampled):
                        out_container.mux(packet)
            for packet in out_stream.encode(None):
                out_container.mux(packet)

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
    """Trim ``audio_path`` into a temp WAV; returns (file, temp_dir)."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ltx_audio_trim_"))
    out_path = temp_dir / "segment.wav"
    trim_audio_start(audio_path, out_path, start_seconds)
    return out_path, temp_dir
