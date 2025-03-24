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
$ docker run --rm -it -v ./tmp/:/app/tmp/ -v huggingface:/root/.cache/huggingface --gpus=all transcriber tmp/<audio_to_transcribe>
```
