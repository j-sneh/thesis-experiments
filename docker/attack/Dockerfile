# Builds GPU docker image of PyTorch
# Uses multi-staged approach to reduce size
# Stage 1
# Use base conda image to reduce time
FROM mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:latest
# Specify py version
ENV PYTHON_VERSION=3.10
# Install apt libs - copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile

ARG OLLAMA_TGZ_URL=https://ollama.com/download/ollama-linux-amd64.tgz
RUN apt-get update && \
    apt-get install -y --no-install-recommends --only-upgrade sqlite3 libsqlite3-0 && \
    apt-get install ffmpeg libsm6 libxext6 -y

# Infiniband packages
RUN apt install -y libibverbs1 ibverbs-providers infiniband-diags

# docker conda env 1
# COPY ./conda-env.yaml ./conda-env.yaml
# RUN conda env create -f conda-env.yaml


# # Create our conda env - copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
RUN conda create --name attack python=${PYTHON_VERSION} ipython jupyter pip

# Below is copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
# We don't install pytorch here yet since CUDA isn't available
# instead we use the direct torch wheel
ENV PATH /opt/conda/envs/vllm_env/bin:$PATH
# Activate our bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

COPY ./requirements.txt ./requirements.txt
RUN conda init
RUN source activate attack
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install -U setuptools
RUN conda install nvidia/label/cuda-12.6.0::cuda-toolkit -c nvidia/label/cuda-12.6.0
RUN export CUDA_HOME=`which nvcc`
RUN pip install -r requirements.txt
RUN pip install flash-attn --no-build-isolation
RUN pip install azureml-sdk --no-cache-dir
# RUN pip install vllm>=0.6.2
RUN pip install -U azureml-fsspec adlfs

RUN echo "source activate attack" >> ~/.profile

# after we build the docker we start the set up for attack conda


# ---- Install Ollama ----
RUN mkdir -p /opt/ollama && \
    curl -fsSL ${OLLAMA_TGZ_URL} -o /tmp/ollama.tgz && \
    tar -C /opt/ollama -xzf /tmp/ollama.tgz && \
    ln -sf /opt/ollama/bin/ollama /usr/local/bin/ollama && \
    rm -f /tmp/ollama.tgz
RUN /usr/local/bin/ollama --version || true

# (Optional) If your scripts expect pkg/bin/ollama to exist:
RUN mkdir -p /workspace/pkg/bin && ln -sf /opt/ollama/bin/ollama /workspace/pkg/bin/ollama

# Activate the virtualenv
CMD ["/bin/bash"]