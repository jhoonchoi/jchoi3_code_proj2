import os


def get_cache_filename(output_file):
    root, _ = os.path.splitext(output_file)
    parent_dir = os.path.dirname(output_file)
    if dir:
        os.makedirs(parent_dir, exist_ok=True)
    return root+".pt"
