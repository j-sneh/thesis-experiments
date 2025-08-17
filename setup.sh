#!/bin/bash
set -euo pipefail
# Set up conda env from conda-env.yaml
if ! conda list --name attack >/dev/null 2>&1; then
	echo "Setting up conda env from conda-env.yaml..."
	conda env create -f conda-env.yaml
else
	echo "conda env already exists"
fi

# if [ ! -f pkg/bin/ollama ]; then
#     echo "Downloading Ollama for Linux (amd64)..."
#     if curl -LO https://ollama.com/download/ollama-linux-amd64.tgz; then
#         echo "Download successful. Extracting Ollama to pkg/..."
#         mkdir -p pkg
#         tar -C pkg/ -xzf ollama-linux-amd64.tgz
#         echo "Extraction complete."
#         rm ollama-linux-amd64.tgz
#     else
#         echo "Download failed. Aborting installation."
#         exit 1
#     fi
# else
#     echo "Ollama already exists at pkg/bin/ollama. Skipping download."
# fi

#!/usr/bin/env bash
set -euo pipefail

# Start the server in the background
pkg/bin/ollama serve &
SERVER_PID=$!

# Always kill server on exit
cleanup() {
	    echo "Stopping Ollama server (PID: $SERVER_PID)..."
	        kill "$SERVER_PID" 2>/dev/null || true
		    wait "$SERVER_PID" 2>/dev/null || true
	    } 
trap cleanup EXIT

# Wait until the server is ready
echo "Waiting for Ollama server to start..."
until pkg/bin/ollama list >/dev/null 2>&1; do
	sleep 1
done
echo "Ollama server is up."

# Pull models
pkg/bin/ollama pull qwen3:8b
pkg/bin/ollama pull llama3.2:3b
pkg/bin/ollama pull phi4-mini:3.8b-fp16
pkg/bin/ollama pull gpt-oss:20b
pkg/bin/ollama create phi4-mini-tool-prompt:3.8b-fp16 -f Modelfiles/phi4.Modelfile

echo "Model downloads complete."
