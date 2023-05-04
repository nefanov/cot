import os
import glob


def get_filenames_in_dir(directory=os.getcwd(), ext=".c"):
    return [f for f in glob.glob(directory + '/**/*'+ext, recursive=True)]


def test_get_filenames_in_dir():
    print(get_filenames_in_dir(ext=".*"))