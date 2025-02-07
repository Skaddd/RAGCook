# RAGCook


This project aims at implementing a simple RAG pipeline based on cooking recipies extracted from [marmitton website](https://www.marmiton.org/).

The code uses `Qdrant` to create a local vectorstore, `Langchain`and `Langgraph`libraries were used to develop the RAG pipeline.
To make the whole code easily usable, a *Dockerfile* is also provided. Finally, the whole code is wrapped into an API using `FastAPI`.

:warning: As `llama-cpp-python` is highly difficult to make work (a lot of building issues) the docker image might not work properly, thus a openai compatibility was added to the project. The steps to use this compatibility is explained later. The current docker will :

- Build `llama-cpp` from source
- Installs `llama-cpp-python`

> NOTE: These actions will take some time

**As the objective was to develop a RAG using local ressources, text pre-processing was not performed using `openai`APIs**.

## Running the project

To run the project and test it, one should check whether docker is installed, for linux check the following [Docker Guide](https://docs.docker.com/engine/install/ubuntu/).

**Before running the project, for simplicity, git-lfs was not added to the docker image, rather one should download the `gguf`model and place it inside `weights/` folder**. 

>NOTE: If you don't download `qwen2.5-1.5b-instruct-fp16.gguf` consider modifying the config file`conf/llm_global_config.yml` folder

Create the docker image then run it
```bash
$ docker build -t cookapi .
$ docker run -p 8000:8000 cookapi
```

Since only the `8000`port was exposed you must map this one, if you want to modify the port, you can directly modify the docker image.

## Testing the project

```python
import requests
rq  =requests.post(url="http://0.0.0.0:8000/chat", json={"user_message" : "Peut-tu me donner les étapes à suivre si je veux cuisiner des aiguillettes de canard"})
rq.content
```

## Repository architecture 

To understand more deeply the project, below the project current architecture
```
RAGCook/
├── conf
│   ├── llm_global_config.yml
│   └── retrieval_pipeline.yml
├── data
│   ├── RAW_HTML
│   └── VECTORSTORE
├── Dockerfile
├── poetry.lock
├── pyproject.toml
├── README.md
├── src
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── utils
│   │   ├── document_processing.py
│   │   ├── helpers.py  
│   │   └── utils_llm.py
│   └── vectorstore.py
└── weights
    └── DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf
```


## Openai compatbility


## Possible Evolutions

A wide range of improvements can be done:

- Implementing test functions
- Largely improve the logger (loguru)
- Implementing evaluation pipeline for both the retrieval part and the generation part
- Transform the Qdrant from local to a service, plus improve the search part. (Adding Hybrid search etc, better indexing)
- Largely improve the RAG workflow to be more efficient and robust (self reflection, Lost in the middle, query rewritting, grading...)
- Try several llm models to asses performances based on the evaluation pipeline