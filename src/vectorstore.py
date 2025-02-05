import logging
import os
from typing import List
from uuid import uuid4

from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger(__name__)


def generate_vectorstore(
    persistent_directory_path: str,
    collection_name: str,
    langchain_documents: List[Document],
    dense_embedding_model,
) -> QdrantVectorStore:
    if not os.path.exists(persistent_directory_path):
        os.mkdir(persistent_directory_path)

    qdrant_client = QdrantClient(path=persistent_directory_path)

    if not qdrant_client.collection_exists(collection_name=collection_name):
        logger.info(
            f" The following collection : {collection_name}"
            + "was not found, creating it.."
        )
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=3072, distance=Distance.COSINE)
            },
        )
    else:
        print("collection already exists !")
        logger.info(
            f" The following collection : {collection_name}"
            + "was found ! taking it back..."
        )
        num_points = qdrant_client.get_collection(
            collection_name=collection_name
        ).points_count
        print(
            "Number of existing point within the vectorstore:"
            + f"{num_points}"
        )

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.DENSE,
        embedding=dense_embedding_model,
        vector_name="dense",
    )
    # to change for something stochastic
    uuids = [str(uuid4()) for _ in range(len(langchain_documents))]

    vectorstore.add_documents(documents=langchain_documents, ids=uuids)

    return vectorstore


def load_vectorstore_as_retriever_tool(
    persistent_directory_path: str,
    collection_name: str,
    retriever_name: str,
    retriever_description: str,
    dense_embedding_model,
) -> Tool:
    qdrant_client = QdrantClient(path=persistent_directory_path)

    if not qdrant_client.collection_exists(collection_name=collection_name):
        logger.warning(f"{collection_name} was not found")
        raise Exception

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.DENSE,
        embedding=dense_embedding_model,
        vector_name="dense",
    )

    retriever_tool = create_retriever_tool(
        retriever=vectorstore.as_retriever(),
        name=retriever_name,
        description=retriever_description,
    )

    return retriever_tool


if __name__ == "__main__":

    from utils.document_processing import process_documents
    from utils.helpers import global_loading_configuration
    from utils.utils_llm import load_dense_embedding_model

    config_dir = r"/home/mateo/projects/RAGCook/conf"

    conf = global_loading_configuration(configuration_dir=config_dir)

    generate_vectorstore(
        persistent_directory_path=conf["vectorstore_persistant_path"],
        collection_name=conf["db_collection_name"],
        dense_embedding_model=load_dense_embedding_model(
            embedding_config=conf["embedding_config"]
        ),
        langchain_documents=process_documents(
            html_folder=conf["saving_html_dir"]
        ),
    )
