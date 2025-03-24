# Running

```sh
$ ./prepare.sh
$ ./run.sh <path_to_audio_file>
```

# Docker / Podman

## Build
```sh
$ docker build -t transcriber -f Dockerfile
```

## Run
```sh
$ docker run --rm -it \
    -v ./tmp/:/app/tmp/ \ # Temporary directory to work within
    -v huggingface:/root/.cache/huggingface \ # Cache of Whisper model
    --gpus=all \ # Allows access to GPU
    transcriber tmp/<audio_to_transcribe> # path to audio & other flags
```
