#!/usr/bin/env python3

from pathlib import Path
import shutil

DATASET_DIRECTORY: Path = Path("./sd04/png_txt")
WORKING_DIRECTORY: Path = Path("./workdir")
TRAIN_DIRECTORY: Path = Path(WORKING_DIRECTORY, "train")
TEST_DIRECTORY: Path = Path(WORKING_DIRECTORY, "test")


def compile_train_directory(data_directory: Path):
    for directory in data_directory.iterdir():
        directory: Path = Path(data_directory, directory)

        for filename in directory.iterdir():
            if filename.name.startswith("Thumbs.db"):
                continue

            number: int = int(filename.name.split("_")[1])
            file_src_path: Path = Path(directory, filename)
            file_dst_path: Path = Path(TRAIN_DIRECTORY, filename)

            # f0001-f1499 & s0001-s1499
            if number < 1500:
                shutil.copyfile(file_src_path, file_dst_path)


def compile_test_directory(data_directory: Path):
    for directory in data_directory.iterdir():
        directory: Path = Path(data_directory, directory)

        for filename in directory.iterdir():
            if filename.name.startswith("Thumbs.db"):
                continue

            number: int = int(filename.name.split("_")[1])
            file_src_path: Path = Path(directory, filename)
            file_dst_path: Path = Path(TEST_DIRECTORY, filename)

            # f1501-f2000 & s1501-s2000
            if number >= 1500:
                shutil.copyfile(file_src_path, file_dst_path)


def main():
    compile_train_directory(DATASET_DIRECTORY)
    compile_test_directory(DATASET_DIRECTORY)


if __name__ == "__main__":
    main()
