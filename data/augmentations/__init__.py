from .auto_augment import AutoAugment, RandAugment
from .augment import RandomDistortedCrop, CenterCrop, Resize
from .augment import RandomDistortColor


AUGS = {
    "AutoAugment": AutoAugment,
    "RandAugment": RandAugment,
    "CenterCrop": CenterCrop,
    "RandomDistortedCrop": RandomDistortedCrop,
    "Resize": Resize,
    "RandomDistortColor": RandomDistortColor
}


class Compose(object):
    def __init__(self, cfgs):
        self.augs = []
        for cfg in cfgs:
            for k, kwargs in cfg.items():
                self.augs.append(AUGS[k](**kwargs))

    def __call__(self, image):
        for aug in self.augs:
            image = aug(image)
        
        return image
