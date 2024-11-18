from enum import Enum


class Gender(Enum):
    FEMALE = "F"
    MALE = "M"


class FingerprintFeature(Enum):
    ARCH = "A"
    LEFT_LOOP = "L"
    RIGHT_LOOP = "R"
    TENTED_ARCH = "T"
    WHIRL = "W"
