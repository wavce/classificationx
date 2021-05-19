from utils.register import Register


DATASETS = Register(name="datasets")


def build_dataset(dataset, **kwargs):
    return DATASETS[dataset](**kwargs).dataset()
