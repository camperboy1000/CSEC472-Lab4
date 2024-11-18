from enum import Enum
from typing import override


class Gender(Enum):
    FEMALE = "F"
    MALE = "M"

    @override
    def __repr__(self) -> str:
        return "{0}.{1}".format(self.__class__.__name__, self.name)

    @override
    def __str__(self) -> str:
        return self.name.title()


class FingerprintFeature(Enum):
    ARCH = "A"
    LEFT_LOOP = "L"
    RIGHT_LOOP = "R"
    TENTED_ARCH = "T"
    WHIRL = "W"

    @override
    def __repr__(self) -> str:
        return "{0}.{1}".format(self.__class__.__name__, self.name)

    @override
    def __str__(self) -> str:
        return self.name.replace("_", " ").title()
