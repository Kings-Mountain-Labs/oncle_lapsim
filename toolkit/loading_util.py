import os

def make_path(part_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", part_path))
