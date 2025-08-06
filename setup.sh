#!/bin/bash

# Set up conda env from conda-env.yaml
echo "Setting up conda env from conda-env.yaml..."
conda env create -f conda-env.yaml

if [ ! -f pkg/bin/ollama ]; then
    echo "Downloading Ollama for Linux (amd64)..."
    if curl -LO https://ollama.com/download/ollama-linux-amd64.tgz; then
        echo "Download successful. Extracting Ollama to pkg/..."
        mkdir -p pkg
        tar -C pkg/ -xzf ollama-linux-amd64.tgz
        echo "Extraction complete."
        rm ollama-linux-amd64.tgz
    else
        echo "Download failed. Aborting installation."
        exit 1
    fi
else
    echo "Ollama already exists at pkg/bin/ollama. Skipping download."
fi