from utils.register import Register


MODELS = Register(name="models")


def build_model(model, **kwargs):
    return MODELS[model](**kwargs)
