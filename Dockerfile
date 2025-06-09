# Dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install Python and essential packages
RUN apt-get update && \
    apt-get install -y python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the model and code
COPY ./Tiny-Llama-1.1B-chat-finetune /app/Tiny-Llama-1.1B-chat-finetune
COPY main.py /app/main.py

RUN mkdir -p /app/offload
WORKDIR /app

# Expose port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
