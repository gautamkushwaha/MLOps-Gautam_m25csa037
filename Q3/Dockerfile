FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY lora_weights /app/lora_weights

# You would copy your Python script here, e.g., train.py or inference.py
# For example:
COPY train_lora.py 

# If you have a specific Python script to run, define your command here.
# For example, to run a training script:
CMD ["python", "train_lora.py"]

# Or, to start a bash shell for interactive use (useful for debugging):
CMD ["bash"]

# Expose ports if your application serves content (e.g., a web API for inference)
# EXPOSE 8000
