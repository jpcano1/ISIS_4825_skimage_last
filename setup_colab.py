import os
import requests

def download_github_content(path):
    dir = path.rsplit("/", 1)[0]
    if not os.path.exists(dir):
        os.makedirs(dir)
        os.system(f"touch {dir}/__init__.py")

    elif not os.path.exists(f"{dir}/__init__.py"):
        os.system(f"touch {dir}/__init__.py")

    os.system(f"shred -u {path}")
    os.system(f"wget https://raw.githubusercontent.com/jpcano1/ISIS_4825_skimage_last/master/{path} -O {path}")

def setup_workshop():
    try:
        download_github_content("utils/visualization.py")
        print("Workshop enabled successfully!")
    except Exception as e:
        raise e