FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 as BASE

WORKDIR /app
ENV PATH="/app/venv/bin:$PATH"

FROM BASE AS build

ENV PYTHONBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /app/venv
RUN python -m pip install --no-cache -r requirements.txt

FROM BASE

COPY --from=build /app/venv ./venv

COPY src/ src/

VOLUME [ "/data" ]

ENTRYPOINT [ "python", "-m", "src" ]