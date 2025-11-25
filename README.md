In this project I wanted to implement a RAG based routing system. It runs 100% locally using Ollama and LlamaIndex.

Current Architecture

Router: Uses Llama3 to analyze the intent of the question.

Data Sources: Distinct indexes for Italian, American, Medieval, Chinese, and Vegetarian cuisines.

Retrieval: Vector search using HuggingFace embeddings (BAAI/bge-small-en-v1.5).

Inference: Generates answers using local Llama3 8B.

Setup

You need to install and start ollama.

Dependencies: streamlit llama-index llama-index-llms-ollama llama-index-embeddings-huggingface ollama

Add Data:
Place your .txt cookbook (or any other) files in the root directory.

Running the App

Option 1: Web Interface (Streamlit)

streamlit run router_interface.py


Option 2: Terminal Mode

python router_terminal.py