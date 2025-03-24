FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 as BASE

WORKDIR /app
ENV PATH="/app/venv/bin:$PATH"

RUN apt-get update && \
    apt-get install git python3-pip python3-venv ffmpeg -y && \
    apt-get clean

FROM BASE AS build

ENV PYTHONBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python3 -m venv /app/venv

COPY requirements.txt requirements.txt

RUN python3 -m pip install --no-cache -r requirements.txt

FROM BASE

COPY --from=build /app/venv ./venv

COPY src/ src/

VOLUME [ "/app/tmp" ]

ENTRYPOINT [ "python", "-m", "src" ]