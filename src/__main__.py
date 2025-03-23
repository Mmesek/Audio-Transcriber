import argparse
import os

from src.utils import process, save, segmentize

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to audio file", default="audio.mp3")
parser.add_argument("-o", "--output", help="Path to output file. Default is <input>.txt")
parser.add_argument("-t", "--tmp", help="Path to output directory. Default is tmp/", default="tmp/")
parser.add_argument("-w", "--word_separation", help="Separate sentences on word-level granularity", default=False)
parser.add_argument("-s", "--single_sentence", help="Merge same-speaker sentences into single one", default=False)
parser.add_argument(
    "-m",
    "--model",
    help="Model to use. Default is `distil-large-v3`",
    default="distil-large-v3",
)

args = parser.parse_args()

print(f"Starting transcription using {args.model} of file {args.input}")


def load():
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cuda", compute_type="int8")
    return model


os.makedirs(args.tmp, exist_ok=True)

model = load()
result = args.output or ".".join(args.input.split(".")[0:-1] + ["txt"])
save(result, segmentize(process(args.input, model, args.word_separation), args.single_sentence))
