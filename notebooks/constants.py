"""Costant values"""

import os

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
DEFAULT_DATA_ROOT = os.getenv("DEFAULT_DATA_ROOT", "../data")
DEFAULT_IMG_DEST = os.getenv("DEFAULT_IMG_DEST", "./results-img")
NOTEBOOKS_CONFIG_FILE = os.getenv("NOTEBOOKS_CONFIG_FILE", "notebooks-config.json")
