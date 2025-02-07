import logging
from typing import Dict, Union

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def llamacpp_load_llm_model(llm_config: Dict[str, Union[str, int]]):
    """Load Langchain chatmodel using llama.cpp.

    Args:
        llm_config (Dict[str, Union[str, int]]): config for the loading.

    Returns:
        BaseChatModel: CHatmodel selected.
    """
    try:
        chat_model = ChatLlamaCpp(
            model_path=llm_config["model_path"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
            n_ctx=llm_config["n_ctx"],
            top_p=llm_config["top_p"],
            verbose=False,
        )
    except KeyError:
        logger.info("Config file missing key parameters to load LLM")

    logger.info("CPP model sucessfully loaded !")

    return chat_model


def llamacpp_load_embedding_model(embedding_config: Dict[str, str]):
    """Load dense embedding model from HF.

    Args:
        embedding_config (Dict[str, str]): config for
        the embedding model.

    Returns:
        Embeddings: Dense embedding object.
    """
    try:
        # dense_embedding_model = SentenceTransformer(
        #     embedding_config["model_name"]
        # )
        embedding_func = HuggingFaceEmbeddings(
            model_name=embedding_config["model_name"]
        )
    except ValueError:
        logger.info(f"Unable to load : {embedding_config['model_name']}")

    logger.info("Embedding HF model sucessfully loaded !")

    return embedding_func


# As i struggled a lot with llama-cpp, openai were used at the begining.


def openai_load_llm_model(
    llm_config: Dict[str, Union[str, int]]
) -> BaseChatModel:
    """Load Langchain chatmodel using openai api.

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


def openai_load_dense_embedding_model(
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


if __name__ == "__main__":

    from helpers import global_loading_configuration

    config_dir = r"/home/mateo/projects/RAGCook/conf"

    conf = global_loading_configuration(configuration_dir=config_dir)

    llamacpp_load_embedding_model(conf["llama_embedding_config"])
