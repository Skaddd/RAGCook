import logging
import os
from typing import List

from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.utils.helpers import generate_hash
from src.utils.utils_llm import llamacpp_load_embedding_model

logger = logging.getLogger(__name__)


def generate_vectorstore(
    persistent_directory_path: str,
    collection_name: str,
    langchain_documents: List[Document],
    dense_emb_model,
    size: int = 384,
) -> QdrantVectorStore:
    """Generate Qdrant local vectorstore.

    Args:
        persistent_directory_path (str): persistent location
        for db.
        collection_name (str): collection name.
        langchain_documents (List[Document]): list of langchain
        documents to feed into the vectorstore.
        dense_emb_model (_type_): Dense embedding model.

    Returns:
        QdrantVectorStore: Vectorstore.
    """
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
                "dense": VectorParams(
                    size=size,
                    distance=Distance.COSINE,
                )
            },
        )
    else:
        logger.info("collection already exists !")
        logger.info(
            f" The following collection : {collection_name}"
            + "was found ! taking it back..."
        )
        num_points = qdrant_client.get_collection(
            collection_name=collection_name
        ).points_count
        logger.info(
            "Number of existing point within the vectorstore:"
            + f"{num_points}"
        )

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.DENSE,
        embedding=dense_emb_model,
        vector_name="dense",
    )
    # Having true unique ids allows qdrant to
    # avoid creating duplicates when happening already existing point.

    unique_ids = [
        str(generate_hash(doc.page_content)) for doc in langchain_documents
    ]

    vectorstore.add_documents(documents=langchain_documents, ids=unique_ids)

    return vectorstore


def load_vectorstore_as_retriever_tool(
    persistent_directory_path: str,
    collection_name: str,
    retriever_name: str,
    retriever_description: str,
    dense_embedding_model,
) -> Tool:
    """Transform vectorstore into retriever tool.

    This function aims at transforming the vectorstore into
    a Langchain Tool, which will make the retrieval proces
    more efficient.
    Args:
        persistent_directory_path (str): persistent directory
        collection_name (str): collection name
        retriever_name (str): retriever name.
        retriever_description (str): retriever description
        used to give information about the retriever tool.
        dense_embedding_model (Embedding): Embedding model.
        use_open_api (bool): Whether to use openai apis.

    Raises:
        Exception: Expected qdrant collection was not found.

    Returns:
        Tool: Retriever as a tool.
    """

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

    config_dir = r"/home/mateo/projects/RAGCook/conf"

    conf = global_loading_configuration(configuration_dir=config_dir)

    generate_vectorstore(
        persistent_directory_path=conf["persistent_dir_path"],
        collection_name=conf["db_collection_name"],
        dense_emb_model=llamacpp_load_embedding_model(
            embedding_config=conf["llama_embedding_config"]
        ),
        langchain_documents=process_documents(
            html_folder=conf["saving_html_dir"],
        ),
    )
