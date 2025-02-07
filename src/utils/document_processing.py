import logging
import os
from glob import glob
from typing import Any, List, Optional, Dict

import re
import requests
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from src.utils.utils_llm import openai_load_llm_model


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


def use_performant_document_parser(
    html_loader_output: Document, llm_config: Dict[str, str]
) -> Document:

    pydantic_parser = PydanticOutputParser(pydantic_object=Recipe)
    recipe_prompt = PromptTemplate(
        template="""
            You are an expert at extracting structured information
            from unstructured text.

            Task:
            Extract the following structured details
            from the provided raw recipe text:

            - **Recipe Title**: The main title of the recipe.
            - **Recipe Process**: A step-by-step guide on how to prepare
            the recipe.
            - **Recipe Ingredients** : List of ingredients for the recipe.

            Output format:
            ```
            {{
                "recipe_title": "<title>",
                "recipe_process": ["step 1", "step 2", "step 3"],
                "ingredients": ["ingredient1", "ingredient2"]
            }}
            ```

            Raw Recipe Text:
            {raw_text}

            Only return the JSON object. Do not add any extra text.
            """,
        input_variables=["raw_text"],
        partial_variables={
            "format_instructions": pydantic_parser.get_format_instructions()
        },
    )
    llm_model = openai_load_llm_model(llm_config=llm_config)

    processed_chain_docs = recipe_prompt | llm_model | pydantic_parser

    extracted_recipe: Recipe = processed_chain_docs.invoke(html_loader_output)

    return Document(
        page_content="\n".join(extracted_recipe.recipe_process),
        metadata={
            "recipe_title": extracted_recipe.recipe_title,
            "ingredients": extracted_recipe.ingredients,
        },
    )


def use_basic_document_parser(html_loader_output: Document) -> Document:
    """Process simply recipe from marmitton website.

    Highly specific function that is not a good practice
    when dealing with large volume of same text.
    Args:
        html_loader_output (Document): RAw document from
        HTMLParser

    Returns:
        Document: processed document based on recipe structure.
    """

    ingredients_pattern = r"Ingrédients\s*(.*?)(?=\s*Ustensiles|Préparation)"
    preparation_pattern = (
        r"Préparation\s*(.*?)(?=\s*Commentaires|Anonyme|Marmiton$)"
    )

    # Search for the ingredients and preparation using regex
    ingredients_match = re.search(
        ingredients_pattern, html_loader_output.page_content, re.DOTALL
    )
    preparation_match = re.search(
        preparation_pattern, html_loader_output.page_content, re.DOTALL
    )
    metadata = html_loader_output.metadata
    # Extracting the data if matches are found
    metadata.update(
        {
            "ingredients": (
                ingredients_match.group(1).strip() if ingredients_match else ""
            )
        }
    )
    preparation = (
        preparation_match.group(1).strip()
        if preparation_match
        else html_loader_output.page_content
    )

    return Document(page_content=preparation, metadata=metadata)


def process_documents(
    html_folder: str,
    llm_config: Dict[str, str] = {},
    use_llm_parser: bool = False,
    file_extension: str = "*.aspx",
) -> List[Document]:
    """Process and chunk raw documents.

    Args:
        html_folder (str): HTML folder containing
        raw document files.
        llm_config (Dict[str,str]): llm parameters.
        use_llm_parse (bool): Whether to improve the
        raw extraction for better chunking.
        file_extension (str, optional): File extension to focus on.
        Defaults to "*.aspx".
    Returns:
        List[Document]: Langchain prepared documents.
    """

    processed_documents = []
    for html_file in glob(
        os.path.join(html_folder, "**", file_extension), recursive=True
    ):
        html_loader = UnstructuredHTMLLoader(html_file, mode="single")
        raw_document = html_loader.load()
        # Higly not working on tiny LLM
        if use_llm_parser:

            processed_documents.append(
                use_performant_document_parser(
                    html_loader_output=raw_document, llm_config=llm_config
                )
            )
        else:
            processed_documents.append(
                use_basic_document_parser(raw_document[-1])
            )

    logger.info(f"Number of chunks : {len(processed_documents)}")

    return processed_documents
