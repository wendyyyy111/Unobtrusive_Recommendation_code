# src/config.py
import os

DATA_DIR = os.environ.get("WESAD_DATA_DIR", "./data/WESAD")

FS_CHEST_DEFAULT = 600
WIN_SEC = 60
STEP_SEC = 30

TOP_K = 3
LAMBDA_LIST = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

ALL_SUBJECTS = [i for i in range(2, 18) if i not in (12,)]

DEFAULT_DROP_PROB = 0.3
DEFAULT_ALPHA = 0.4
DEFAULT_LAMBDA = 0.5
DEFAULT_MU = 0.5
DEFAULT_SEQ_LEN = 5