import requests
from typing import List, Any
import os
from langchain_community.document_loaders import UnstructuredHTMLLoader
from glob import glob


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


def process_documents(html_folder: str, file_extension: str = "*.aspx"):

    raw_documents = []
    for html_file in glob(
        os.path.join(html_folder, "**", file_extension), recursive=True
    ):
        html_loader = UnstructuredHTMLLoader(html_file, mode="single")
        raw_documents.extend(html_loader.load())

    return raw_documents


if __name__ == "__main__":

    saving_dir = "./data"
    download_reciepes(
        list_urls=[],
        saving_dir=saving_dir,
    )
