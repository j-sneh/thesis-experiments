#!/bin/bash
set -euo pipefail
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
pkg/bin/ollama pull qwen2.5:7b
pkg/bin/ollama pull llama3.1:8b
pkg/bin/ollama pull gpt-oss:20b
echo "Model downloads complete."
