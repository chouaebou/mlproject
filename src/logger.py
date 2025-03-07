import os
import logging
from datetime import datetime

DEFAULT_PATH = os.getcwd()
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

build_logs_path = os.path.join(DEFAULT_PATH, "logs", LOG_FILE)
os.makedirs(build_logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(build_logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)