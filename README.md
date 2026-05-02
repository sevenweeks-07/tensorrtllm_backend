# TensorRT-LLM Backend for Qwen2-VL

This repository provides the TensorRT-LLM backend for Triton Inference Server, specifically configured to run the Qwen2-VL multimodal model. Qwen2-VL is a powerful vision-language model that can process both text and images for tasks like image captioning, visual question answering, and more.

## Prerequisites

- A system with an NVIDIA GPU (preferably with at least 16GB VRAM for Qwen2-VL-7B)
- Docker installed and configured
- Git and Git LFS installed
- Access to Hugging Face models (you may need to log in with `huggingface-cli login` if required)

## Installation

### 1. Clone the Repository

Clone the TensorRT-LLM backend repository and initialize the submodules:

```bash
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend
git lfs install
git submodule update --init --recursive
```

### 2. Install NVIDIA Container Toolkit

To enable GPU support in Docker containers, install the NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3. Start the Triton Server Docker Container

Launch the Docker container with the necessary configurations:

```bash
docker run --rm -ti \
  --net=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/mnt \
  -w /mnt \
  --gpus all \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 bash
```

### 4. Install Python Requirements

Install the required Python packages for Qwen2-VL:

```bash
pip install -r ./tensorrt_llm/triton_backend/all_models/multimodal/requirements-qwen2vl.txt
```

## Building the Model Engine

### 1. Download the Model

Clone the Qwen2-VL model from Hugging Face:

```bash
export MODEL_NAME="Qwen2-VL-7B-Instruct"
git clone https://huggingface.co/Qwen/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
```

### 2. Set Environment Variables

Define the paths for the model and engines:

```bash
export PMIX_MCA_gds=hash
export MODEL_NAME="Qwen2-VL-7B-Instruct"
export UNIFIED_CKPT_PATH=tmp/trt_models/${MODEL_NAME}/fp16/1-gpu
export HF_MODEL_PATH=tmp/hf_models/${MODEL_NAME}
export ENGINE_PATH=tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu
export MULTIMODAL_ENGINE_PATH=tmp/trt_engines/${MODEL_NAME}/multimodal_encoder
export ENCODER_INPUT_FEATURES_DTYPE=TYPE_FP16
```

### 3. Convert the Checkpoint

Convert the Hugging Face model to TensorRT-LLM format:

```bash
python3 ./tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
  --model_dir ${HF_MODEL_PATH} \
  --output_dir ${UNIFIED_CKPT_PATH} \
  --dtype float16
```

### 4. Build the TensorRT Engine

Build the optimized TensorRT engine for the language model:

```bash
trtllm-build \
  --checkpoint_dir ${UNIFIED_CKPT_PATH} \
  --output_dir ${ENGINE_PATH} \
  --gemm_plugin=float16 \
  --gpt_attention_plugin=float16 \
  --max_batch_size 4 \
  --max_input_len 2048 \
  --max_seq_len 3072 \
  --max_multimodal_len 1296
```

### 5. Build the Multimodal Engine

Build the engine for the vision encoder:

```bash
python3 ./tensorrt_llm/examples/models/core/multimodal/build_multimodal_engine.py \
  --model_type qwen2_vl \
  --model_path tmp/hf_models/${MODEL_NAME} \
  --output_dir ${MULTIMODAL_ENGINE_PATH}
```

## Preparing Triton Configurations

Set up the Triton model repository with the necessary configurations:

```bash
# Copy the base model repository
cp tensorrt_llm/triton_backend/all_models/inflight_batcher_llm/ multimodal_ifb -r

# Copy multimodal-specific configurations
cp tensorrt_llm/triton_backend/all_models/multimodal/ensemble multimodal_ifb -r
cp tensorrt_llm/triton_backend/all_models/multimodal/multimodal_encoders multimodal_ifb -r

# Fill in the configuration templates
python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/tensorrt_llm/config.pbtxt \
  triton_backend:tensorrtllm,triton_max_batch_size:4,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},logits_datatype:TYPE_FP32,prompt_embedding_table_data_type:TYPE_FP16,cross_kv_cache_fraction:0.5

python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/preprocessing/config.pbtxt \
  tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:4,preprocessing_instance_count:1,multimodal_model_path:${MULTIMODAL_ENGINE_PATH},engine_dir:${ENGINE_PATH},max_num_images:1,max_queue_delay_microseconds:20000

python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/postprocessing/config.pbtxt \
  tokenizer_dir:${HF_MODEL_PATH},triton_max_batch_size:4,postprocessing_instance_count:1

python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/ensemble/config.pbtxt \
  triton_max_batch_size:4,logits_datatype:TYPE_FP32

python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/tensorrt_llm_bls/config.pbtxt \
  triton_max_batch_size:4,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:tensorrt_llm,multimodal_encoders_name:multimodal_encoders,logits_datatype:TYPE_FP32,prompt_embedding_table_data_type:TYPE_FP16

python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/multimodal_encoders/config.pbtxt \
  triton_max_batch_size:4,multimodal_model_path:${MULTIMODAL_ENGINE_PATH},encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},hf_model_path:${HF_MODEL_PATH},max_queue_delay_microseconds:20000
```

## Launching the Server

Install the Triton client and launch the server:

```bash
pip install tritonclient[grpc]

export PMIX_MCA_gds=hash

python3 tensorrt_llm/triton_backend/scripts/launch_triton_server.py \
  --world_size 1 \
  --model_repo=multimodal_ifb/ \
  --tensorrt_llm_model_name tensorrt_llm,multimodal_encoders \
  --multimodal_gpu0_cuda_mem_pool_bytes 300000000
```

## Usage

Once the server is running, you can send requests to the model. Here are some examples:

### Basic Text and Image Query

```bash
python tensorrt_llm/triton_backend/tools/multimodal/client.py \
  --text "Describe this image:" \
  --image "path/to/your/image.jpg" \
  --request-output-len 100 \
  --model_type qwen2_vl
```

### Streaming Response

```bash
python tensorrt_llm/triton_backend/tools/multimodal/client.py \
  --text "What do you see in this image?" \
  --image "path/to/your/image.jpg" \
  --request-output-len 100 \
  --model_type qwen2_vl \
  --streaming
```

### Using BLS (Business Logic Scripting)

```bash
python tensorrt_llm/triton_backend/tools/multimodal/client.py \
  --text "Analyze this image:" \
  --image "path/to/your/image.jpg" \
  --request-output-len 100 \
  --model_type qwen2_vl \
  --use_bls
```

## Notes

- Adjust `max_batch_size`, `max_multimodal_len`, and other parameters based on your GPU memory and requirements.
- For multi-image requests, increase `max_num_images` in the preprocessing config.
- If you encounter MPI-related errors, ensure `PMIX_MCA_gds=hash` is set.
- The multimodal engine handles vision encoding, while the main engine processes text generation.
- For production deployment, consider using the `tensorrtllm_backend` Docker image instead of the development container.

## Troubleshooting

- Ensure all paths are correct and models are properly downloaded.
- Check GPU memory usage; Qwen2-VL models require significant VRAM.
- Verify that the Docker container has access to the GPU.
- If builds fail, check the TensorRT-LLM documentation for compatibility.

For more detailed information, refer to the [TensorRT-LLM documentation](https://github.com/NVIDIA/TensorRT-LLM) and [Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html).

