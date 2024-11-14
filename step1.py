import fingerprint_feature_extractor
import fingerprint_enhancer
import logging
import cv2
import os

logger = logging.getLogger(__name__)

# Process the TRAIN set to extract features (identify minutiae)

# Calculate minutiae for each reference image (f_i) and its corresponding subject image s_i.
# You'll generally compare distances between these images

# Try three different image matching or ML technqiues to do this


def load_and_enhance_image(directory, filename):
    # Each image is 512 x 512 pixels with 32 rows of white space at the bottom
    # Classified using one of the five following classes: A=Arch, L=Left Loop, R=Right Loop, T=Tented Arch, W=Whorl
    # Each text file for the image has the gender, class, and history information

    image_path = os.path.join(directory, filename)

    # Read input image
    img = cv2.imread(image_path, 0)

    # Enhance fingerprint image
    enhanced_img = fingerprint_enhancer.enhance_fingerprint(img)

    logger.info(f"Loaded and enhanced image at {image_path}")

    return enhanced_img


def minutiae_extraction(directory, filename):
    image_path = os.path.join(directory, filename)

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
