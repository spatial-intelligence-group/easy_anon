"""Utilities for the easy-anon package."""

import importlib.resources
from appdirs import user_cache_dir
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
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
