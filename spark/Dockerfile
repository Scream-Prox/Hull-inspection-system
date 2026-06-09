FROM spark:4.0.1-python3

USER root

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    pyarrow \
    pillow \
    opencv-python-headless \
    torch==2.5.1 \
    torchvision==0.20.1

USER spark
