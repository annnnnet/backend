import os
import shutil
import sys
from datetime import datetime

def get_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

