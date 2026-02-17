# HuggingFace Docker Assignment

Model Link:
https://huggingface.co/Gautam20/distilbert-review-genres

Training:
docker build -t hf-train:v1 .
docker run -it --rm --gpus all hf-train:v1

Evaluation:
docker build -t hf-eval:v1 -f Dockerfile.eval .
docker run -it --rm --gpus all hf-eval:v1
