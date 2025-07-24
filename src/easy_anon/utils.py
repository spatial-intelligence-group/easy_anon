"""Utilities for the easy-anon package."""

import os
import importlib.resources
from appdirs import user_cache_dir
import yaml
import numpy as np
import cv2
from urllib.request import urlopen
from functools import partial
import uuid
from rich.console import Console
from rich.theme import Theme
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".npy", ".npz"]
MODELS_FILE = "configs/models.yaml"


def load_model_list():
    """Load the list of available Mask2Former models from YAML file.

    Args:
        None

    Returns:
        dict: A dictionary with model names as keys and their configurations as values. The configuration includes
        the relative path to model config file (in Mask2Former configs directory) and the model weights URL.
    """
    with importlib.resources.files("easy_anon").joinpath(MODELS_FILE).open("r") as f:
        model_list_raw = yaml.load(f, Loader=Loader)
    model_list = {f"{d}-{m}": model_list_raw[d][m] for d in model_list_raw for m in model_list_raw[d]}
    return model_list


MODELS = load_model_list()
LABELS_CHOICES = ["animal", "person", "sky", "snow", "vegetation", "vehicle", "water"]
LABELS_FILES = {
    "ADE20k": "configs/labels_ade20k.yaml",
    "Cityscapes": "configs/labels_cityscapes.yaml",
    "MapillaryVistas": "configs/labels_mapillary_vistas.yaml",
}

DEFAULT_CACHE_DIR = user_cache_dir("easy-anon")


def check_img_ext(filename):
    """Check if the file has a valid image extension.

    Args:
        filename (str): The name or path of the file to check.

    Returns:
        bool: True if the file has a valid image extension, False otherwise.
    """
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def check_mask_ext(filename):
    """Check if the file has a valid mask extension.

    Args:
        filename (str): The name or path of the file to check.

    Returns:
        bool: True if the file has a valid mask extension, False otherwise.
    """
    return os.path.splitext(filename)[1].lower() in MASK_EXTS


def load_mask(mask_path, mode="black_on_white", decision_threshold=0.5):
    """Load the mask and convert it to the desired color mode.

    Args:
        mask_path (str): Path to the mask file.
        mode (str): Color mode of the mask. Options are:
            - "black_on_white": Black regions will be anonymized.
            - "white_on_black": White regions will be anonymized.
        decision_threshold (float): Binarization threshold for masks stored as images. Default is 0.5.

    Returns:
        np.ndarray: The binary mask where the regions to be anonymized are set to True.
    """
    if os.path.splitext(mask_path)[1].lower() == ".npy":
        mask = np.load(mask_path)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    elif os.path.splitext(mask_path)[1].lower() == ".npz":
        mask = np.load(mask_path)["mask"]
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        mask = mask > decision_threshold

    if mask is None:
        raise ValueError(f"Could not read the mask file: {mask_path}")

    if mode == "black_on_white":
        mask = mask == 0
    elif mode == "white_on_black":
        mask = mask == 1

    return mask


def save_mask(mask, mask_path, mode="black_on_white"):
    """Save the mask to a file.

    Args:
        mask (np.ndarray): The binary mask to save.
        mask_path (str): Path where the mask will be saved.
        mode (str): Color mode of the mask. Options are:
            - "black_on_white": Black regions will be anonymized.
            - "white_on_black": White regions will be anonymized.

    Returns:
        None
    """
    if mode == "black_on_white":
        mask = mask == 0

    if os.path.splitext(mask_path)[1].lower() == ".npy":
        np.save(mask_path, mask)
    elif os.path.splitext(mask_path)[1].lower() == ".npz":
        np.savez(mask_path, mask=mask)
    else:
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])


def get_rich_theme():
    """Get a Rich theme with a custom CLI style.

    Args:
        None

    Returns:
        rich.theme.Theme: A Rich Theme object with custom CLI style.
    """
    return Theme(
        styles={
            "bar.back": "#9B9B9B",
            "bar.complete": "#0065BD",
            "bar.finished": "#6AADE4",
            "bar.pulse": "#F0AB00",
            "log.message": "default",
            "warning": "#F0AB00",
        },
        inherit=False,
    )


def get_rich_argparse_style():
    """Get a Rich style for rich-argparse formatter.

    Args:
        None

    Returns:
        dict: A dictionary with Rich style settings for rich-argparse.
    """
    return {
        "argparse.args": "#0065BD",
        "argparse.groups": "#F0AB00",
        "argparse.help": "default",
        "argparse.metavar": "#9B9B9B",
        "argparse.prog": "#9B9B9B",
        "argparse.syntax": "bold",
        "argparse.text": "default",
        "argparse.default": "#6AADE4 italic",
    }


def get_rich_console():
    """Create a Rich Console for logging and output.

    Args:
        None

    Returns:
        rich.console.Console: A Rich Console object with custom theme and settings.
    """
    return Console(
        theme=get_rich_theme(),
        width=None,  # Use terminal width
        log_time=False,  # Disable automatic timestamps
        log_path=False,  # Disable automatic log path
    )


def get_rich_progress_processing():
    """Create a progress bar for processing tasks.

    Args:
        None

    Returns:
        rich.progress.Progress: A Rich Progress object configured for processing tasks.
    """
    return Progress(
        TextColumn("[progress.description]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        "•",
        TimeRemainingColumn(),
        console=get_rich_console(),
    )


def get_rich_progress_download():
    """Create a progress bar for downloading tasks.

    Args:
        None

    Returns:
        rich.progress.Progress: A Rich Progress object configured for download tasks.
    """
    return Progress(
        TextColumn("Downloading {task.fields[file_name]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        console=get_rich_console(),
    )


def download_checkpoint(url: str, dest: str):
    """Download a model checkpoint from the given URL to the specified destination.

    Args:
        url (str): The URL to download the checkpoint from.
        dest (str): The destination path where the checkpoint will be saved.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    progress = get_rich_progress_download()
    with progress:
        name = url.split("/")[-1]
        task = progress.add_task("download", file_name=name, start=False)
        response = urlopen(url)
        progress.update(task, total=int(response.info()["Content-length"]))
        with open(dest, "wb") as dest_file:
            progress.start_task(task)
            for data in iter(partial(response.read, 32768), b""):
                dest_file.write(data)
                progress.update(task, advance=len(data))
        progress.console.log(f"Downloaded {dest}")


def get_unique_string():
    """Generate a random unique string.

    Args:
        None

    Returns:
        str: A random unique string
    """
    return str(uuid.uuid4()).replace("-", "")
