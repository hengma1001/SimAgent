import os
import uuid
from datetime import datetime


def get_work_dir(tag="tag"):
    timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
    work_dir = os.path.abspath(f"{tag}-{timestamp}-{uuid.uuid4()}")
    os.makedirs(work_dir)
    return work_dir
