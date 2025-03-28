import argparse
import os

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

args = parser.parse_args()


def load():
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cuda", compute_type="int8")
    return model


def main(filepath: str, model, speakers: dict[str, str]):
    result = args.output or ".".join(filepath.split(".")[0:-1] + ["txt"])
    print(f"Starting transcription using {args.model} of file {filepath} to {result}")

    save(
        result,
        segmentize(process(filepath, model, args.word_separation, args.tmp, speakers), args.single_sentence),
        args.metadata,
    )


def get_files(filepath: str):
    files = []
    for file in filepath:
        if os.path.isdir(file):
            for file in os.listdir(file):
                files.append(file)
        else:
            files.append(file)
    return files


if __name__ == "__main__":
    os.makedirs(args.tmp, exist_ok=True)
    model = load()
    speakers = {"stream_1": "Speaker", "stream_2": "Mic"}

    for file in get_files(args.input):
        main(file, model, speakers)
