import argparse
import os
from pathlib import Path

from src.utils import process, save, segmentize

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to audio file", nargs="*", default="audio.mp3")
parser.add_argument("-o", "--output", help="Path to output file. Default is <input>.txt. Works for single audio file")
parser.add_argument("-t", "--tmp", help="Path to output directory. Default is tmp/", default="tmp/")
parser.add_argument(
    "-w", "--word_separation", help="Separate sentences on word-level granularity", default=False, action="store_true"
)
parser.add_argument(
    "-s", "--single_sentence", help="Merge same-speaker sentences into single one", default=False, action="store_true"
)
parser.add_argument("-M", "--metadata", help="Whether to include timestamp/speaker", default=False, action="store_true")
parser.add_argument(
    "-m",
    "--model",
    help="Model to use. Default is `distil-large-v3`",
    default="distil-large-v3",
)
parser.add_argument("-l", "--language", help="Language of the file", default="en")

args = parser.parse_args()


def load():
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cuda", compute_type="int8")
    return model


def main(filepath: Path, model, speakers: dict[str, str]):
    result = args.output or Path(filepath.parent, ".".join(filepath.name.split(".")[0:-1] + ["txt"]))
    print(f"Starting transcription using {args.model} of file {filepath} to {result}")

    save(
        result,
        segmentize(
            process(str(filepath), model, args.word_separation, args.tmp, speakers, args.language), args.single_sentence
        ),
        args.metadata,
    )


def get_files(filepath: str):
    files = []
    for path in filepath:
        if os.path.isdir(path):
            for file in os.listdir(path):
                files.append(Path(path, file))
        else:
            files.append(Path(path))
    return files


if __name__ == "__main__":
    os.makedirs(args.tmp, exist_ok=True)
    model = load()
    speakers = {"stream_1": "Speaker", "stream_2": "Mic"}

    for file in get_files(args.input):
        main(file, model, speakers)
