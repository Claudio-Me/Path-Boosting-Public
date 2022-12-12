from enum import Enum, auto


class ModelType(Enum):
    '''Enum class for regressor types'''

    r_model = auto()
    xgb_one_step = auto()
