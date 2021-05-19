from .base_config import Config
from .efficientnet_config import get_efficientnet_config


CONFIG_DICT = {
    "EfficientNetB0": get_efficientnet_config("EfficientNetB0"),
    "EfficientNetB1": get_efficientnet_config("EfficientNetB1"),
    "EfficientNetB2": get_efficientnet_config("EfficientNetB2"),
    "EfficientNetB3": get_efficientnet_config("EfficientNetB3"),
    "EfficientNetB4": get_efficientnet_config("EfficientNetB4"),
    "EfficientNetB5": get_efficientnet_config("EfficientNetB5"),
    "EfficientNetB6": get_efficientnet_config("EfficientNetB6"),
    "EfficientNetB7": get_efficientnet_config("EfficientNetB7"),
}


def build_configs(name):
    return CONFIG_DICT[name]
