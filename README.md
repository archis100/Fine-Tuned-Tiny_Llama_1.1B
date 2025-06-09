# Fine-Tuned-Tiny_Llama_1.1B

# Create conda environment:
conda create --name tiny_llama_env python=3.11 -y

# Activate the environment
conda activate tiny_llama_env

# Install PyTorch (since bitsandbytes requires it)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install the other dependencies
pip install -q git+https://github.com/huggingface/transformers peft accelerate bitsandbytes safetensors sentencepiece trl

# Install FastAPI & Uvicorn
pip install fastapi uvicorn

# Check Installation
pip list | findstr "transformers peft accelerate bitsandbytes safetensors sentencepiece trl fastapi uvicorn"

# Build and run the Docker container in the terminal
cd project-folder/
docker build -t tiny_llama-api .
docker run --gpus all -p 8000:8000 tiny_llama-api