#!/usr/bin/env python3

import logging
import shutil
from pathlib import Path
from time import sleep

TRAINING_DIR_NAME: str = "train"
TEST_DIR_NAME: str = "test"
TRAINING_RANGE: range = range(0, 1500)
TEST_RANGE: range = range(1500, 2001)


def setup_work_directory(dataset_directory: Path, work_directory: Path) -> None:
    """Setup the work directory by organizing and copying files from the datset
    directory. This should be called prior to any other function call to ensure
    the work directory is properly set up.
    """

    # Raise an exception if the dataset directory doesn't exist
    if not dataset_directory.is_dir():
        raise FileNotFoundError(f"No such directory: {dataset_directory}")

    # Define the training and test directories
    training_directory = work_directory.joinpath(TRAINING_DIR_NAME)
    test_directory = work_directory.joinpath(TEST_DIR_NAME)
    logging.info(f"Copying data from {dataset_directory} to {work_directory}...")

    # Ensure the work directory exists and is empty
    if work_directory.is_file():
        raise FileExistsError(f"Found a file: {work_directory}")
    elif work_directory.is_dir():
        logging.warning(f"{work_directory} already exists, removing in 10 seconds...")
        sleep(10)

        # Nuke the work directory from orbit
        shutil.rmtree(work_directory)

    # Create the work tree
    training_directory.mkdir(parents=True)
    test_directory.mkdir(parents=True)

    # Walk the dataset directory for the fingerprint files
    for current_dir, _, file_list in dataset_directory.walk():
        for file_name in file_list:
            # If not a png or txt, we don't want it
            dataset_path = current_dir.joinpath(file_name)
            if dataset_path.suffix not in (".png", ".txt"):
                logging.info(f"Skipping file: {dataset_path}")
                continue

            # Extract the number from the filename
            file_number: int = int(file_name[1:5])

            # Figure out the dest path
            work_path: Path
            if file_number in TRAINING_RANGE:
                work_path = training_directory.joinpath(file_name)
            elif file_number in TEST_RANGE:
                work_path = test_directory.joinpath(file_name)
            else:
                # If we get here, something is wrong with the dataset
                raise ValueError(
                    f"File {file_name} does not fall in either the training or testing range!"
                )

            # Copy the file to its now home
            logging.debug(f"Copying {dataset_path} to {work_path}")
            shutil.copy(dataset_path, work_path)


def main():
    logging.basicConfig()

    dataset_directory = Path.cwd().joinpath("sd04", "png_txt")
    work_directory = Path.cwd().joinpath("workdir")

    setup_work_directory(dataset_directory, work_directory)


if __name__ == "__main__":
    main()
