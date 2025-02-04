from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from typing import Dict, Union
import logging


logger = logging.getLogger(__name__)


def load_llm_model(llm_config: Dict[str, Union[str, int]]) -> BaseChatModel:
    """Load Langchain chatmodel.

    Args:
        llm_config (Dict[str, Union[str, int]]): config for the loading.

    Returns:
        BaseChatModel: CHatmodel selected.
    """

    try:
        chat_model = ChatOpenAI(
            api_key=llm_config["api_key"],
            model=llm_config["model_name"],
            temperature=llm_config["temperature"],
            streaming=llm_config["streaming"],
        )
    except KeyError:
        logger.info("Config file missing key parameters to load LLM")

    return chat_model


def load_dense_embedding_model(
    embedding_config: Dict[str, Union[str, int]]
) -> Embeddings:
    """Load dense embedding model.

    Args:
        embedding_config (Dict[str, Union[str, int]]): config for
        the embedding model.

    Returns:
        Embeddings: Dense embedding object.
    """

    try:
        dense_embedding_model = OpenAIEmbeddings(
            model=embedding_config["model_name"],
            api_key=embedding_config["api_key"],
        )
    except KeyError:
        print("error")
        logger.info("Config file missing key parameters to load Embedding")

    return dense_embedding_model
