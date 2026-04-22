FROM spark:4.0.1-python3 AS spark

FROM apache/nifi:1.28.1

USER root

RUN apt-get update && \
    apt-get install -y openjdk-17-jdk python3 python3-pip && \
    pip3 install --no-cache-dir numpy pandas pyarrow pillow opencv-python-headless torch==2.5.1 torchvision==0.20.1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=spark /opt/spark /opt/spark

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PATH=${JAVA_HOME}/bin:${SPARK_HOME}/bin:${PATH}

USER nifi
