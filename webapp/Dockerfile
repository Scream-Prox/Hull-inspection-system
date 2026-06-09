FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY entrypoint.sh .

RUN chmod +x /app/entrypoint.sh

EXPOSE 8501
EXPOSE 8502

CMD ["/bin/sh", "/app/entrypoint.sh"]
