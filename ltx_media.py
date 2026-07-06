"""
PyAV-backed audio/video helpers for ltx-ws.

All media trimming, segmentation, concat, muxing, and inference audio loading
goes through this module (PyAV / ``pip install av``).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from fractions import Fraction
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


def _pyav_frame_rate(fps: float | int | Fraction) -> Fraction:
    """Coerce fps to ``Fraction`` for PyAV ``add_stream`` (plain float breaks)."""
    if isinstance(fps, Fraction):
        return fps
    if isinstance(fps, int):
        return Fraction(fps, 1)
    fps_f = float(fps)
    if fps_f == int(fps_f):
        return Fraction(int(fps_f), 1)
    return Fraction(round(fps_f * 1000), 1000).limit_denominator(1001)


def _add_remux_stream(
    out_container: Any,
    in_stream: Any,
) -> Any:
    """Create an output stream for remuxing compressed packets from ``in_stream``."""
    add_from_template = getattr(out_container, "add_stream_from_template", None)
    if callable(add_from_template):
        return add_from_template(in_stream)
    try:
        return out_container.add_stream(template=in_stream)
    except TypeError as exc:
        raise RuntimeError(
            "PyAV remux requires add_stream_from_template (upgrade: pip install 'av>=12')"
        ) from exc


def _media_time_seconds(obj: Any, stream: Any | None = None) -> float | None:
    """Best-effort timestamp in seconds for PyAV packets/frames across versions."""
    t = getattr(obj, "time", None)
    if t is not None:
        return float(t)
    pts = getattr(obj, "pts", None)
    if pts is None:
        return None
    if stream is not None and stream.time_base is not None:
        return float(pts * stream.time_base)
    time_base = getattr(obj, "time_base", None)
    if time_base is not None:
        return float(pts * time_base)
    return None


def media_available() -> bool:
    """True when PyAV is importable."""
    return _AV_AVAILABLE


def require_media() -> None:
    if not _AV_AVAILABLE:
        raise RuntimeError("PyAV is required — install with: pip install av")


def probe_video_info(video_path: str) -> Any:
    """PyAV replacement for ``ltx_core_mlx.utils.ffmpeg.probe_video_info``."""
    try:
        from ltx_core_mlx.utils.ffmpeg import VideoInfo
    except ImportError as exc:  # pragma: no cover - only when ltx-core-mlx missing
        raise RuntimeError("ltx-core-mlx is required for video probing") from exc

    require_media()
    with av.open(str(video_path)) as container:
        video_stream = None
        has_audio = False
        for stream in container.streams:
            if stream.type == "video" and video_stream is None:
                video_stream = stream
            elif stream.type == "audio":
                has_audio = True
        if video_stream is None:
            raise RuntimeError(f"No video stream found in {video_path}")

        width = int(video_stream.width or 0)
        height = int(video_stream.height or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions in {video_path}")

        rate = video_stream.average_rate or video_stream.base_rate
        fps = float(rate) if rate else 24.0

        duration = 0.0
        if container.duration:
            duration = float(container.duration) / float(av.time_base)
        elif video_stream.duration is not None and video_stream.time_base is not None:
            duration = float(video_stream.duration * video_stream.time_base)

        num_frames = int(video_stream.frames) if video_stream.frames else 0
        if num_frames == 0 and duration > 0 and fps > 0:
            num_frames = int(duration * fps)

        if num_frames == 0:
            counted = 0
            for _ in container.decode(video_stream):
                counted += 1
            num_frames = counted

        return VideoInfo(
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            duration=duration,
            has_audio=has_audio,
        )


def load_video_frames_normalized(
    path: str,
    height: int,
    width: int,
    max_frames: int,
    fps: float | None = None,
) -> Any:
    """PyAV replacement for ``ltx_core_mlx.utils.video.load_video_frames_normalized``."""
    import mlx.core as mx
    import numpy as np

    require_media()
    frames_list: list[Any] = []
    next_pick = 0.0
    decoded_index = 0

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream found in {path}")
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or stream.base_rate or 24.0)

        for frame in container.decode(stream):
            if len(frames_list) >= max_frames:
                break

            take = True
            if fps is not None and source_fps > 0 and abs(source_fps - fps) > 0.01:
                take = decoded_index >= next_pick
                if take:
                    next_pick += source_fps / fps

            if not take:
                decoded_index += 1
                continue

            rgb = frame.reformat(width=width, height=height, format="rgb24")
            arr = np.asarray(rgb.to_ndarray(), dtype=np.float32) / 255.0
            frames_list.append(arr)
            decoded_index += 1

    if not frames_list:
        raise RuntimeError(f"No frames decoded from {path}")

    stacked = np.stack(frames_list, axis=0)
    tensor = mx.array(stacked).transpose(0, 3, 1, 2)
    tensor = tensor.transpose(1, 0, 2, 3)[None, ...]
    return tensor.astype(mx.bfloat16)


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
                    out_video = _add_remux_stream(out_container, in_video)
                    if in_audio is not None:
                        out_audio = _add_remux_stream(out_container, in_audio)
                elif in_audio is not None and out_audio is None:
                    out_audio = _add_remux_stream(out_container, in_audio)

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
        out_v = out_container.add_stream(
            "libx265", rate=_pyav_frame_rate(rate), width=width, height=height
        )
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
            v_out = _add_remux_stream(out, v_in)
            a_out = out.add_stream("aac", rate=AUDIO_OUTPUT_RATE, layout=AUDIO_OUTPUT_LAYOUT)

            for packet in vin.demux(v_in):
                if packet.dts is None and packet.pts is None:
                    continue
                packet_time = _media_time_seconds(packet, v_in)
                if packet_time is not None and packet_time > duration_s:
                    break
                packet.stream = v_out
                out.mux(packet)

            for frame in ain.decode(a_in):
                frame_time = _media_time_seconds(frame, a_in)
                if frame_time is not None and frame_time > duration_s:
                    break
                for resampled in resampler.resample(frame):
                    for packet in a_out.encode(resampled):
                        out.mux(packet)
            for packet in a_out.encode(None):
                out.mux(packet)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Audio mux produced empty output: {output_path}")
    return output_path


def extract_audio_from_video(
    video_path: Path | str,
    output_path: Path | str,
    *,
    duration_s: float | None = None,
) -> bool:
    """Extract the audio track from ``video_path`` to AAC in ``output_path``.

    Returns ``False`` when the source has no audio stream.
    """
    require_media()
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(video_path)) as vin:
        if not vin.streams.audio:
            return False
        a_in = vin.streams.audio[0]

        with av.open(str(output_path), "w") as out:
            a_out = out.add_stream("aac", rate=AUDIO_OUTPUT_RATE, layout=AUDIO_OUTPUT_LAYOUT)
            resampler = AudioResampler(
                format="fltp",
                layout=AUDIO_OUTPUT_LAYOUT,
                rate=AUDIO_OUTPUT_RATE,
            )
            for frame in vin.decode(a_in):
                frame_time = _media_time_seconds(frame, a_in)
                if duration_s is not None and frame_time is not None and frame_time > duration_s:
                    break
                for resampled in resampler.resample(frame):
                    for packet in a_out.encode(resampled):
                        out.mux(packet)
            for packet in a_out.encode(None):
                out.mux(packet)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        return False
    return True


def replace_output_audio_from_source(
    generated_video: Path | str,
    audio_source_video: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """Keep generated visuals; replace audio with the source clip's audio track."""
    generated_video = Path(generated_video)
    audio_source_video = Path(audio_source_video)
    if output_path is None:
        output_path = generated_video
    else:
        output_path = Path(output_path)

    info = probe_video_info(generated_video)
    duration_s = max(0.1, float(info.duration or 0.0))
    if duration_s <= 0.1 and info.num_frames > 0 and info.fps > 0:
        duration_s = max(0.1, (info.num_frames - 1) / info.fps)

    tmp_audio = output_path.with_suffix(".source_audio.m4a")
    try:
        if not extract_audio_from_video(audio_source_video, tmp_audio, duration_s=duration_s):
            if output_path != generated_video:
                shutil.copy2(generated_video, output_path)
            return Path(output_path)
        tmp_out = output_path.with_suffix(".remux.mp4") if output_path == generated_video else output_path
        mux_audio_into_video(generated_video, tmp_audio, tmp_out, duration_s=duration_s)
        if tmp_out != output_path:
            tmp_out.replace(output_path)
        return Path(output_path)
    finally:
        tmp_audio.unlink(missing_ok=True)


def stream_decoder_latent_to_mp4(
    decoder: Any,
    latent: Any,
    output_path: Path | str,
    *,
    frame_rate: float,
    audio_path: Path | str | None = None,
) -> Path:
    """Decode VAE latent tiles to H.264 MP4 via PyAV (replaces upstream ffmpeg pipe)."""
    import mlx.core as mx
    import numpy as np

    require_media()
    from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling
    from ltx_core_mlx.utils.memory import aggressive_cleanup

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_rate = float(frame_rate)
    fps_frac = _pyav_frame_rate(frame_rate)
    tiling = _compute_decode_tiling(latent.shape, frame_rate=frame_rate)
    _, _, _f_lat, h_lat, w_lat = latent.shape
    out_h = int(h_lat * 32)
    out_w = int(w_lat * 32)
    expected_frames = max(1, int(latent.shape[2] * 8 - 7))

    video_tmp = output_path
    cleanup_video_tmp = False
    if audio_path:
        from ltx_paths import mk_scratch_file

        fd, tmp = mk_scratch_file("ltx_vid_", ".mp4")
        os.close(fd)
        video_tmp = Path(tmp)
        cleanup_video_tmp = True

    frames_written = 0
    with av.open(str(video_tmp), "w") as container:
        stream = container.add_stream("libx264", rate=fps_frac, width=out_w, height=out_h)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        stream.time_base = Fraction(fps_frac.denominator, fps_frac.numerator)

        for chunk in decoder.tiled_decode(latent, tiling):
            num_frames = int(chunk.shape[2])
            for i in range(num_frames):
                frame = chunk[:, :, i, :, :]
                frame = mx.clip(frame, -1.0, 1.0)
                frame = ((frame + 1.0) * 127.5).astype(mx.uint8)
                frame_hwc = frame[0].transpose(1, 2, 0)
                mx.eval(frame_hwc)
                rgb = np.asarray(frame_hwc)
                video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                video_frame = video_frame.reformat(format="yuv420p")
                video_frame.pts = frames_written
                for packet in stream.encode(video_frame):
                    container.mux(packet)
                frames_written += 1
                if i % 8 == 0:
                    aggressive_cleanup()
            del chunk
            aggressive_cleanup()
        for packet in stream.encode(None):
            container.mux(packet)

    aggressive_cleanup()

    if frames_written <= 0:
        raise RuntimeError(
            f"VAE decode wrote 0 video frames (expected ~{expected_frames})"
        )
    if not video_tmp.is_file() or video_tmp.stat().st_size == 0:
        raise RuntimeError(f"VAE decode produced empty video: {video_tmp}")

    duration_s = frames_written / frame_rate
    if audio_path:
        mux_audio_into_video(video_tmp, audio_path, output_path, duration_s=duration_s)
        if cleanup_video_tmp:
            video_tmp.unlink(missing_ok=True)
    elif video_tmp != output_path:
        shutil.move(str(video_tmp), str(output_path))

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"MP4 encode produced empty output: {output_path}")
    return output_path


def encode_image_hold_video(
    image_path: str | Path,
    output_path: str | Path,
    *,
    width: int,
    height: int,
    num_frames: int,
    fps: float = 24.0,
) -> Path:
    """Encode a still image as an H.264 clip with ``num_frames`` identical frames.

    IC-LoRA ``images`` at frame 0 only replace the first latent frame; per-frame
    identity across a clip requires ``video_conditioning`` with a hold clip built
    from the character still (see ltx-2-mlx IC-LoRA static-scene recipes).
    """
    import numpy as np
    from PIL import Image

    require_media()
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = int(width)
    height = int(height)
    num_frames = max(1, int(num_frames))
    fps_frac = _pyav_frame_rate(fps)

    with Image.open(image_path) as im:
        rgb = im.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        arr = np.asarray(rgb, dtype=np.uint8)

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    pad_w = width + (width & 1)
    pad_h = height + (height & 1)
    if (pad_w, pad_h) != (width, height):
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        padded[:height, :width, :] = arr
        arr = padded

    with av.open(str(output_path), "w") as container:
        stream = container.add_stream(
            "libx264", rate=fps_frac, width=pad_w, height=pad_h
        )
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        stream.time_base = Fraction(fps_frac.denominator, fps_frac.numerator)
        for i in range(num_frames):
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Image hold video encode produced empty output: {output_path}")
    return output_path


def trim_video_to_spec(
    src: str | Path,
    dst: str | Path,
    *,
    num_frames: int,
    width: int,
    height: int,
    fps: float,
    start_seconds: float = 0.0,
) -> Path:
    """Re-encode up to ``num_frames`` from ``src`` at the target resolution and fps."""
    import numpy as np

    require_media()
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    num_frames = max(1, int(num_frames))
    width = int(width)
    height = int(height)
    pad_w = width + (width & 1)
    pad_h = height + (height & 1)
    fps_frac = _pyav_frame_rate(fps)
    start_seconds = max(0.0, float(start_seconds))

    frames_written = 0
    next_pick = 0.0
    decoded_index = 0
    started = start_seconds <= 0.0
    last_arr: Any | None = None

    with av.open(str(src)) as vin, av.open(str(dst), "w") as out:
        if not vin.streams.video:
            raise RuntimeError(f"No video stream in {src}")
        in_stream = vin.streams.video[0]
        source_fps = float(in_stream.average_rate or in_stream.base_rate or fps)

        out_stream = out.add_stream(
            "libx264", rate=fps_frac, width=pad_w, height=pad_h
        )
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": "18", "preset": "veryfast"}
        out_stream.time_base = Fraction(fps_frac.denominator, fps_frac.numerator)

        for frame in vin.decode(in_stream):
            if frames_written >= num_frames:
                break

            frame_time = _media_time_seconds(frame, in_stream)
            if not started:
                if frame_time is not None and frame_time < start_seconds:
                    decoded_index += 1
                    continue
                started = True

            take = True
            if source_fps > 0 and abs(source_fps - fps) > 0.01:
                take = decoded_index >= next_pick
                if take:
                    next_pick += source_fps / fps

            if not take:
                decoded_index += 1
                continue

            rgb = frame.reformat(width=pad_w, height=pad_h, format="rgb24")
            arr = np.asarray(rgb.to_ndarray(), dtype=np.uint8)
            last_arr = arr
            out_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            out_frame = out_frame.reformat(format="yuv420p")
            out_frame.pts = frames_written
            for packet in out_stream.encode(out_frame):
                out.mux(packet)
            frames_written += 1
            decoded_index += 1

        if frames_written == 0:
            raise RuntimeError(f"No frames written trimming {src}")

        while frames_written < num_frames and last_arr is not None:
            out_frame = av.VideoFrame.from_ndarray(last_arr, format="rgb24")
            out_frame = out_frame.reformat(format="yuv420p")
            out_frame.pts = frames_written
            for packet in out_stream.encode(out_frame):
                out.mux(packet)
            frames_written += 1

        for packet in out_stream.encode(None):
            out.mux(packet)

    if not dst.is_file() or dst.stat().st_size == 0:
        raise RuntimeError(f"Video trim produced empty output: {dst}")
    return dst


def count_video_frames(path: str | Path) -> int:
    """Count decoded video frames (more reliable than container metadata alone)."""
    require_media()
    path = Path(path)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream in {path}")
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        return sum(1 for _ in container.decode(stream))


def ic_lora_vae_compatible_frame_count(
    num_frames: int,
    *,
    source_num_frames: int | None = None,
) -> int:
    """Match ``ltx_pipelines_mlx.iclora_utils.append_ic_lora_reference_video_conditionings``."""
    max_frames = max(1, int(num_frames))
    if source_num_frames is not None:
        max_frames = min(max_frames, max(1, int(source_num_frames)))
    k = max(1, (max_frames - 1) // 8)
    return 1 + k * 8


def normalize_video_for_ic_lora_reference(
    src: str | Path,
    dst: str | Path,
    *,
    num_frames: int,
    width: int,
    height: int,
    fps: float,
) -> int:
    """Re-encode ``src`` to exactly ``vae_frames`` (1+8k), padding with the last frame."""
    import numpy as np

    require_media()
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    width = int(width)
    height = int(height)
    pad_w = width + (width & 1)
    pad_h = height + (height & 1)
    fps_frac = _pyav_frame_rate(fps)
    target_frames = max(1, int(num_frames))
    decoded: list[Any] = []

    with av.open(str(src)) as vin:
        if not vin.streams.video:
            raise RuntimeError(f"No video stream in {src}")
        stream = vin.streams.video[0]
        for frame in vin.decode(stream):
            if len(decoded) >= target_frames:
                break
            rgb = frame.reformat(width=pad_w, height=pad_h, format="rgb24")
            decoded.append(np.asarray(rgb.to_ndarray(), dtype=np.uint8))

    if not decoded:
        raise RuntimeError(f"No frames decoded from {src}")

    source_count = len(decoded)
    vae_frames = ic_lora_vae_compatible_frame_count(
        target_frames,
        source_num_frames=source_count,
    )
    while len(decoded) < vae_frames:
        decoded.append(decoded[-1].copy())

    with av.open(str(dst), "w") as out:
        out_stream = out.add_stream(
            "libx264", rate=fps_frac, width=pad_w, height=pad_h
        )
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": "18", "preset": "veryfast"}
        out_stream.time_base = Fraction(fps_frac.denominator, fps_frac.numerator)
        for idx, arr in enumerate(decoded[:vae_frames]):
            out_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            out_frame = out_frame.reformat(format="yuv420p")
            out_frame.pts = idx
            for packet in out_stream.encode(out_frame):
                out.mux(packet)
        for packet in out_stream.encode(None):
            out.mux(packet)

    if not dst.is_file() or dst.stat().st_size == 0:
        raise RuntimeError(f"IC-LoRA reference normalize produced empty output: {dst}")
    return vae_frames


def encode_single_frame(
    output_file: Any,
    image_array: Any,
    crf: float,
) -> None:
    """Encode one RGB frame to a 1-frame H.264 MP4 via PyAV (I2V preprocess)."""
    import numpy as np

    require_media()
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(
            f"encode_single_frame expects HxWx3 RGB, got {image_array.shape}"
        )

    height, width, _ = image_array.shape
    pad_w = width + (width & 1)
    pad_h = height + (height & 1)
    if (pad_w, pad_h) != (width, height):
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        padded[:height, :width, :] = image_array
        image_array = padded

    from io import BytesIO

    if isinstance(output_file, BytesIO):
        output_file.seek(0)
        output_file.truncate(0)
        container = av.open(output_file, mode="w", format="mp4")
    else:
        container = av.open(str(output_file), mode="w")

    with container:
        stream = container.add_stream(
            "libx264", rate=1, width=pad_w, height=pad_h
        )
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(int(crf)), "preset": "veryfast"}
        frame = av.VideoFrame.from_ndarray(image_array, format="rgb24")
        frame = frame.reformat(format="yuv420p")
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def decode_single_frame(video_file: Any) -> Any:
    """Decode the first frame of an H.264 MP4 buffer/file back to HxWx3 RGB."""
    import numpy as np

    require_media()
    from io import BytesIO

    if isinstance(video_file, BytesIO):
        video_file.seek(0)
        inp: Any = video_file
    else:
        inp = str(video_file)

    with av.open(inp, mode="r") as container:
        if not container.streams.video:
            raise RuntimeError("decode_single_frame: no video stream")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            return np.asarray(frame.reformat(format="rgb24").to_ndarray()).copy()
    raise RuntimeError("decode_single_frame: no frames decoded")


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
