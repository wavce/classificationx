from utils.register import Register


LOSSES = Register(name="losses")

OPTIMIZERS = Register(name="optimizers")

LR_SCHEDULERS = Register(name="lr_schedulers")

METRICS = Register(name="metrics")


def build_loss(loss, **kwargs):
    return LOSSES[loss](**kwargs)


def build_learning_rate_scheduler(scheduler, **kwargs):
    return LR_SCHEDULERS[scheduler](**kwargs)


def build_metric(metric, **kwargs):
    return METRICS[metric](**kwargs)


def build_optimizer(optimizer, **kwargs):
    return OPTIMIZERS[optimizer](**kwargs)
    