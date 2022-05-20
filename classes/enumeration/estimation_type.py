from enum import Enum, auto


class EstimationType(Enum):
    '''Enum class for regressor types'''

    regression = auto()
    classification = auto()