# Dockerfile
# Custom Spark image for the real-time fraud detection prototype.
# Intended to be built from D:\ThucTap_VinSmartFuture\run_realtime
#
# This Dockerfile keeps Spark's official image and adds the Python
# dependencies needed by the streaming/model inference applications.

FROM apache/spark:4.0.1

USER root

RUN python3 -m pip install --no-cache-dir \
    pandas \
    numpy \
    psycopg2-binary \
    torch

RUN mkdir -p /opt/spark-apps \
    /opt/spark-models \
    /opt/spark-data/features \
    /opt/spark-data/history \
    /tmp/checkpoint \
    /tmp/.ivy2

WORKDIR /opt/spark-apps

# Copy these only if they exist in the build context.
# If your docker-compose.yml already mounts these directories,
# the mounts will take precedence at runtime.
COPY spark-apps/ /opt/spark-apps/
COPY models/ /opt/spark-models/

RUN chmod -R 755 /opt/spark-apps /opt/spark-models /opt/spark-data /tmp/checkpoint /tmp/.ivy2

USER spark
