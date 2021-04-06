import os
import shutil
import tempfile
import zipfile

import filelock
import requests
import tqdm

from .logger import logger
from ..settings import CACHE_DIR

# Hide an error message from `tokenizers` if this process is forked.
os.environ["TOKENIZERS_PARALLELISM"] = "True"


def textflint_url(uri):
    return "http://textflint.oss-cn-beijing.aliyuncs.com/download/" + uri


def path_in_cache(file_path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(file_path):
        return file_path
    full_path = os.path.join(CACHE_DIR, file_path)
    if full_path.endswith(".zip"):
        return os.path.dirname(full_path)
    return full_path


def unzip_file(path_to_zip_file, unzipped_folder_path):
    """
    Unzips a .zip file to folder path.

    """
    logger.info(
        f"Unzipping file {path_to_zip_file} to {unzipped_folder_path}.")
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(unzipped_folder_path)


def set_cache_dir(cache_dir):
    """
    Sets all relevant cache directories to ``TR_CACHE_DIR``.

    """
    # Tensorflow Hub cache directory
    os.environ["TFHUB_CACHE_DIR"] = cache_dir
    # HuggingFace `transformers` cache directory
    os.environ["PYTORCH_TRANSFORMERS_CACHE"] = os.path.join(
        cache_dir, "transformers")
    # HuggingFace `datasets` cache directory
    os.environ["HF_HOME"] = cache_dir
    # Basic directory for Linux user-specific non-data files
    os.environ["XDG_CACHE_HOME"] = cache_dir


def download_if_needed(folder_name):
    r"""
    Folder name will be saved as `.cache/textflint/[folder_name]`. If it
    doesn't exist on disk, the zip file will be downloaded and extracted.

    :param str folder_name: path to folder or file in cache
    :return: path to the downloaded folder or file on disk
    """

    cache_dest_path = path_in_cache(folder_name)

    os.makedirs(os.path.dirname(cache_dest_path), exist_ok=True)
    # Use a lock to prevent concurrent downloads.
    cache_dest_lock_path = cache_dest_path + ".lock"
    cache_file_lock = filelock.FileLock(cache_dest_lock_path)
    cache_file_lock.acquire()

    # Check if already downloaded.
    if os.path.exists(cache_dest_path):
        cache_file_lock.release()
        return cache_dest_path
    # If the file isn't found yet, download the zip file to the cache.
    downloaded_file = tempfile.NamedTemporaryFile(
        dir=CACHE_DIR, delete=False
    )
    http_get(folder_name, downloaded_file)

    # Move or unzip the file.
    downloaded_file.close()
    if zipfile.is_zipfile(downloaded_file.name):
        unzip_file(downloaded_file.name, cache_dest_path)
    else:
        logger.info(f"Copying {downloaded_file.name} to {cache_dest_path}.")
        shutil.copyfile(downloaded_file.name, cache_dest_path)
    cache_file_lock.release()

    # Remove the temporary file.
    os.remove(downloaded_file.name)
    logger.info(f"Successfully saved {folder_name} to cache.")

    return cache_dest_path


def http_get(folder_name, out_file, proxies=None):
    """
    Get contents of a URL and save to a file.

    https://github.com/huggingface/transformers/blob/master/src
    /transformers/file_utils.py

    """
    folder_url = textflint_url(folder_name)
    logger.info(f"Downloading {folder_url}.")
    req = requests.get(folder_url, stream=True, proxies=proxies)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    if req.status_code == 403:  # Not found on AWS
        raise Exception(f"Could not find {folder_name} on server.")
    progress = tqdm.tqdm(unit="B", unit_scale=True, total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            out_file.write(chunk)
    progress.close()


if "TR_CACHE_DIR" in os.environ:
    set_cache_dir(os.environ["TR_CACHE_DIR"])
