from dataclasses import dataclass
from faster_whisper import WhisperModel
from src.split_channels import split, detect_audio_sources


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str

    def timestamp(self):
        return f"[{self.start:>7.2f}s -> {self.end:>7.2f}s]"

    def with_speaker(self):
        return f"{self.speaker:>7}:{self.text}\n"

    def with_timestamp(self):
        return f"{self.timestamp()} {self.with_speaker()}"


def transcibe(filepath: str, model: WhisperModel, speaker: str = None, word_separation: bool = False) -> list[Segment]:
    """Transcribes audio file into list of words and timestamps"""
    segments, info = model.transcribe(
        filepath,
        log_progress=True,
        beam_size=10,
        best_of=10,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 1000},
        no_speech_threshold=1,
        temperature=1,
    )

    lines = []

    for segment in segments:
        if word_separation:
            for word in segment.words:
                lines.append(Segment(word.start, word.end, word.word, speaker))
        else:
            lines.append(Segment(segment.start, segment.end, segment.text, speaker))
    return lines


def process(
    filepath: str, model: WhisperModel, word_separation: bool, tmp_path: str, names: dict[str, str]
) -> list[Segment]:
    """Parse audio file. Optionally, splits & extracts separate audio sources from a file"""
    if len(audio := detect_audio_sources(filepath)) > 1:
        print("Detected multiple audio sources, splitting")
        paths = split(filepath, audio, tmp_path)
    else:
        paths = [filepath]
    print("Combining audio sources into single transcript")
    total = []
    for path in paths:
        print("Parsing audio source", path)
        total.extend(transcibe(path, model, names.get(path, "Other"), word_separation))
    return total


def segmentize(total: list[Segment], single_sentence: bool) -> list[Segment]:
    """Combines words into sentences according to timestamp they were first spoken at by speakers"""
    total: list[Segment] = sorted(total, key=lambda x: (x.start, x.end))
    if not single_sentence:
        return total

    final = []
    current_segment = Segment(0, 0, "", "")
    for segment in total:
        if current_segment.speaker != segment.speaker:
            final.append(current_segment)
            current_segment = Segment(segment.start, segment.end, segment.text, segment.speaker)
        else:
            current_segment.text += segment.text
            current_segment.end = segment.end
    final.append(current_segment)
    return final


def save(filepath: str, segments: list[Segment]):
    with open(filepath, "w", newline="", encoding="utf-8") as file:
        file.writelines([i.with_timestamp() for i in segments if i.text])
