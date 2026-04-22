# Use a Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir transformers torch sacrebleu sentencepiece

# Command to run the translation script
CMD ["python", "translate.py"]