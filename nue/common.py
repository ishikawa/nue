import pathlib

BUILD_DIR = pathlib.Path(__file__).parent.parent / "build"
CACHE_DIR = pathlib.Path(__file__).parent.parent / "cache"
DATASET_CACHE_DIR = CACHE_DIR / "datasets"
