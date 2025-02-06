import logging
import os
from glob import glob
from typing import Any, List, Optional

import requests
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils_llm import load_llm_model


class Recipe(BaseModel):
    recipe_title: str = Field(description="title of the recipe")
    recipe_process: List[str] = Field(
        description=(
            "List of step by step guide to realize"
            + " the recipe with explanation"
        )
    )
    ingredients: Optional[List[str]] = Field(
        description="List of ingredients that are necessary for the recipe",
        default=[],
    )


logger = logging.getLogger(__name__)


def save_html_file(
    request_reponse: dict[str, Any], filename: str, saving_dir: str
) -> None:
    """Saving HTML content.

    Args:
        request_reponse (dict[str, Any]): response body.
        filename (str): saving filename.
        saving_dir (str): saving dir location.
    """

    with open(os.path.join(saving_dir, filename), "wb") as ff:
        ff.write(request_reponse.content)


def download_reciepes(list_urls: List[str], saving_dir: str) -> None:
    """Fetch content from list of HTML urls.

    Args:
        list_urls (List[str]): list of html urls to download.
        saving_dir (str): saving location.
    """

    for recipe_url in list_urls:
        try:
            response = requests.get(recipe_url)
            if response.status_code == 200:
                filename = os.path.basename(recipe_url)
                save_html_file(
                    request_reponse=response,
                    filename=filename,
                    saving_dir=saving_dir,
                )

        except requests.RequestException as e:
            print(f"Bad request, check urls : {e}")


def process_documents(
    html_folder: str, llm_config, file_extension: str = "*.aspx"
) -> List[Document]:
    """Process and chunk raw documents.

    Args:
        html_folder (str): HTML folder containing
        raw document files.
        llm_config (_type_): llm parameters.
        file_extension (str, optional): File extension to focus on.
        Defaults to "*.aspx".
    Returns:
        List[Document]: Langchain prepared documents.
    """

    pydantic_parser = PydanticOutputParser(pydantic_object=Recipe)
    recipe_prompt = PromptTemplate(
        template="""
    You are given an unstructured recipe text extracted from a website.
    Your task is to extract:
      - The recipe title
      - The recipe process: List of step by step guide explaining
      how to realise the recipe.
      - List of ingredients to use
    from the following text : {raw_text}
    """,
        input_variables=["raw_text"],
        partial_variables={
            "format_instructions": pydantic_parser.get_format_instructions()
        },
    )

    llm_model = load_llm_model(llm_config=llm_config)

    processed_documents = []
    for html_file in glob(
        os.path.join(html_folder, "**", file_extension), recursive=True
    ):
        html_loader = UnstructuredHTMLLoader(html_file, mode="single")
        processed_chain_docs = recipe_prompt | llm_model | pydantic_parser

        # As we applied specific parser we obtain a Recipe obj
        extracted_recipe: Recipe = processed_chain_docs.invoke(
            html_loader.load()
        )
        processed_documents.add(
            Document(
                page_content=extracted_recipe.recipe_process,
                metadata={
                    "recipe_title": extracted_recipe.recipe_title,
                    "ingredients": extracted_recipe.ingredients,
                },
            )
        )

    logger.info(f"Number of chunks : {len(processed_documents)}")

    return processed_documents


if __name__ == "__main__":

    from helpers import global_loading_configuration

    config_dir = r"/home/mateo/projects/RAGCook/conf"

    conf = global_loading_configuration(configuration_dir=config_dir)

    # download_reciepes(
    #     list_urls=conf["marmitton_urls"],
    #     saving_dir=conf["saving_html_dir"],
    # )
    process_documents(
        html_folder=conf["saving_html_dir"],
        llm_config=conf["llm_config"]["chat_config"],
    )[0]
