#!/usr/bin/env python3
# pyright: reportMissingTypeStubs=false

import logging
import os
from pathlib import Path
from typing import final, override

import cv2
import fingerprint_enhancer
import fingerprint_feature_extractor

from enums import FingerprintFeature, Gender

logger = logging.getLogger(__name__)

# Process the TRAIN set to extract features (identify minutiae)

# Calculate minutiae for each reference image (f_i) and its corresponding subject image s_i.
# You'll generally compare distances between these images

# Try three different image matching or ML technqiues to do this


@final
class FingerprintPair(object):
    number: int
    reference_image: Path
    subject_image: Path
    gender: Gender
    feature: FingerprintFeature

    __slots__ = ("number", "reference_image", "subject_image", "gender", "feature")

    def __init__(
        self,
        number: int,
        reference_image: str | Path,
        subject_image: str | Path,
        gender: str | Gender,
        feature: str | FingerprintFeature,
    ) -> None:
        self.number = number
        self.reference_image = Path(reference_image)
        self.subject_image = Path(subject_image)
        self.gender = Gender(gender)
        self.feature = FingerprintFeature(feature)

    @override
    def __repr__(self) -> str:
        return "FingerprintPair({0},{1},{2},{3},{4})".format(
            repr(self.number),
            repr(self.reference_image),
            repr(self.subject_image),
            repr(self.gender),
            repr(self.feature),
        )

    @override
    def __str__(self) -> str:
        return "Fingerprint Pair: {0} (Gender: {1}, Class: {2})".format(
            self.number, self.gender, self.feature
        )


@final
class PotentialFingerprintPair(object):
    reference_image: Path
    comparision_image: Path
    match: bool

    __slots__ = ("reference_image", "comparision_image", "match")

    def __init__(
        self, reference_image: Path | str, comparision_image: Path | str, match: bool
    ) -> None:
        self.reference_image = Path(reference_image)
        self.comparision_image = Path(comparision_image)
        self.match = match

    @override
    def __repr__(self) -> str:
        return "PotentialFingerprintPari({0},{1},{2})".format(
            repr(self.reference_image),
            repr(self.comparision_image),
            repr(self.match),
        )

    @override
    def __str__(self) -> str:
        return "Potential Fingerprint Pair: {0} & {1} - {2}".format(
            self.reference_image.name, self.comparision_image.name, self.match
        )


def load_dataset(dataset_directory: Path) -> list[FingerprintPair]:
    logging.info(f"Loading dataset from {dataset_directory}")
    dataset: list[FingerprintPair] = []

    for reference_path in dataset_directory.glob("f*.png"):
        # Set the Path for the cooresponding subject image
        number = int(reference_path.name[1:5])
        subject_name = "s" + reference_path.name[1:]
        subject_path = reference_path.with_name(subject_name)
        metadata_path = reference_path.with_suffix(".txt")

        # Set these to None as we don't know them yet but need them later
        gender: Gender = None  # pyright: ignore[reportAssignmentType]
        feature: FingerprintFeature = None  # pyright: ignore[reportAssignmentType]

        # Read the metadata file
        logging.debug(f"Reading {metadata_path}")
        with metadata_path.open() as metadata:
            line = metadata.readline()
            while line != "":
                line_elements = line.split()
                match line_elements[0]:
                    case "Gender:":
                        gender = Gender(line_elements[1])
                    case "Class:":
                        feature = FingerprintFeature(line_elements[1])
                    case "History:":
                        pass
                    case _:
                        logging.warning(
                            f"Found extra line {line} while parsing {metadata_path}"
                        )
                line = metadata.readline()

        # Create the FingerprintPair object
        fingerprint_pair = FingerprintPair(
            number, reference_path, subject_path, gender, feature
        )

        # Add the FingerprintPair to the dataset
        dataset.append(fingerprint_pair)

    return dataset


def load_and_enhance_image(image_path: Path):
    # Each image is 512 x 512 pixels with 32 rows of white space at the bottom
    # Classified using one of the five following classes: A=Arch, L=Left Loop, R=Right Loop, T=Tented Arch, W=Whorl
    # Each text file for the image has the gender, class, and history information

    # Read input image
    image_path_str = os.path.abspath(image_path)
    image = cv2.imread(image_path_str)

    # Enhance fingerprint image
    enhanced_image = fingerprint_enhancer.enhance_fingerprint(image)

    logger.info(f"Loaded and enhanced image at {image_path}")

    return enhanced_image


def minutiae_extraction(image_path: Path):
    print("Minutiae extracted")


def technique1():
    print("Technique 1")


def technique2():
    print("Technique 2")


def technique3():
    print("Technique 3")


# For each method, document your max, min, and average false reject and false accept rates and calculate
# your equal error rate


def main():
    logging.basicConfig(level=logging.INFO)

    training_directory = Path.cwd().joinpath("workdir", "train")
    training_dataset = load_dataset(training_directory)

    for fingerprint_pair in training_dataset:
        print(fingerprint_pair)


if __name__ == "__main__":
    main()
