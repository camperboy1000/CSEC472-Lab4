#!/usr/bin/env python3

import logging
import os
from pathlib import Path
from typing import final

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
    reference_image: Path
    subject_image: Path
    gender: Gender
    feature: FingerprintFeature

    __slots__ = ("reference_image", "subject_image", "gender", "feature")

    def __init__(
        self,
        reference_image: str | Path,
        subject_image: str | Path,
        gender: str | Gender,
        feature: str | FingerprintFeature,
    ) -> None:
        self.reference_image = Path(reference_image)
        self.subject_image = Path(subject_image)
        self.gender = Gender(gender)
        self.feature = FingerprintFeature(feature)


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


def minutiae_extraction(iamge_path: Path):
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
    logging.basicConfig(filename="step1.log", level=logging.INFO)


if __name__ == "__main__":
    main()
