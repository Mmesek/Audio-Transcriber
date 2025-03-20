import ffmpeg
from ffprobe import ffprobe


def detect_audio_sources(filepath: str) -> ffprobe.FFStream:
    """Fetches audio streams metadata from file"""
    return ffprobe.FFProbe(filepath).audio


def split(filepath: str, audio_streams: list[ffprobe.FFStream], tmp_path: str) -> list[str]:
    """Extracts audio streams to multiple files"""
    worker = ffmpeg.FFmpeg().option("y").input(filepath)
    outputs = []
    for stream in audio_streams:
        outputs.append(f"stream_{stream.index}")
        worker.output(
            f"{tmp_path}/stream_{stream.index}.wav",
            map=f"0:a:{int(stream.index) - 1}",
            options={"c:a": "mp3"},
        )
    print("Calling FFmpeg with:", worker.arguments)
    worker.execute()
    return outputs
