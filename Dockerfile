FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.gpu.txt .
RUN pip install --no-cache-dir -r requirements.gpu.txt

COPY . .

CMD ["python", "src/train.py"]
