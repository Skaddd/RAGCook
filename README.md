# RAGCook
___


This project aims at exploring the rapid and concise deployment of RAG pipeline wrapped around `FastAPI` and easily deployable thanks to simple `Docker` implementation.


## Running the project

To run the project and test it, one should check whether docker is installed, for linux check the following [Docker Guide](https://docs.docker.com/engine/install/ubuntu/).

Create the docker image then run it
```bash
$ docker build -t cookapi .
$ docker run -p 8000:8000 cookapi
```

Since only the `8000`port was exposed you must map this one, if you want to modify the port, you can directly modify the docker image.



## Repository architecture 

To understand more deeply the project, below the project current architecture

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



# Possible Evolutions