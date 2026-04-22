FROM spark:4.0.1-python3

USER root

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    pyarrow \
    pillow \
    opencv-python-headless \
    torch==2.5.1 \
    torchvision==0.20.1

USER spark
