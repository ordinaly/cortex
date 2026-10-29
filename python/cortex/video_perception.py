"""Bounded local-video ingestion through ffmpeg and a supplied perception model."""
from pathlib import Path
import subprocess

from .perception import PerceptionAdapter, PerceptionProvider


def video_frames(path, *, sample_fps=2, max_frames=120, width=320, height=240):
    """Yield RGB arrays and sample timestamps; resizing can change aspect ratio.

    Sampling timestamps refer to the resampled stream starting at zero, not
    original presentation timestamps. Use a dedicated decoder for precise timing.
    """
    import numpy as np
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('local video file required')
    if not 0 < sample_fps <= 60 or not 1 <= max_frames <= 10000 or not 1 <= width <= 2048 or not 1 <= height <= 2048:
        raise ValueError('invalid video bounds')
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-an',
               '-vf', f'fps={sample_fps},scale={width}:{height}', '-frames:v', str(max_frames),
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = width * height * 3
    try:
        for i in range(max_frames):
            raw = process.stdout.read(size)
            if not raw:
                break
            if len(raw) != size:
                raise RuntimeError('truncated decoded video frame')
            yield i / sample_fps, np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        if process.wait(timeout=30) != 0:
            raise RuntimeError('video decoder failed')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def ingest_video(path, provider: PerceptionProvider, adapter: PerceptionAdapter, **decode_options):
    source = str(Path(path).resolve())
    reads = []
    for index, (timestamp, image) in enumerate(video_frames(path, **decode_options)):
        frame_id = f'video:{index}'
        frame = provider.perceive(image, frame_id=frame_id, timestamp=timestamp, source=source)
        if (frame.frame_id, frame.timestamp, frame.source) != (frame_id, timestamp, source):
            raise ValueError('provider changed requested frame provenance')
        reads.append(adapter.observe(frame))
    return tuple(reads)
