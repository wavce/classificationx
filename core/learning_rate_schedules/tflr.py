import tensorflow as tf 
from ..builder import LR_SCHEDULERS


class Warmup(object):
    def __init__(self, warmup_steps, start_learning_rate, end_learning_rate):
        self.warmup_steps = warmup_steps
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.learning_rate_increment = (end_learning_rate - start_learning_rate) / warmup_steps

        self.decay_fn = lambda x: x
    
    def __call__(self, step):
        with tf.name_scope("learning_rate_scheduler"):
            with tf.name_scope("warmup"):
                if step < self.warmup_steps:
                    return tf.cast(step, tf.float32) * self.learning_rate_increment + self.start_learning_rate
            
            return self.decay_fn(step - self.warmup_steps)
    

@LR_SCHEDULERS.register
class PiecewiseConstantDecay(Warmup):
    def __init__(self,
                  boundaries,
                  values,
                  warmup_learning_rate=0.,
                  warmup_steps=0,
                  name=None):
        super(PiecewiseConstantDecay, self).__init__(warmup_steps=warmup_steps,
                                                     start_learning_rate=warmup_learning_rate,
                                                     end_learning_rate=values[0])
        self.decay_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries,
                                                                             values=values,
                                                                             name=name)
    def __call__(self, step):
        with tf.name_scope("learning_rate_scheduler"):
            with tf.name_scope("warmup"):
                if step < self.warmup_steps:
                    return tf.cast(step, tf.float32) * self.learning_rate_increment + self.start_learning_rate
            
            return self.decay_fn(step - self.warmup_steps)
        

@LR_SCHEDULERS.register
class PolynomialDecay(Warmup):
    """A LearningRateSchedule that uses a polynomial decay schedule."""

    def __init__(self,
                 initial_learning_rate,
                 train_steps,
                 end_learning_rate=0.0001,
                 power=1.0,
                 cycle=False,
                 warmup_learning_rate=0.,
                 warmup_steps=0,
                 name=None):
        super(PolynomialDecay, self).__init__(warmup_steps=warmup_steps,
                                              start_learning_rate=warmup_learning_rate,
                                              end_learning_rate=initial_learning_rate)
        self.decay_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=initial_learning_rate,
                                                                      decay_steps=train_steps,
                                                                      end_learning_rate=end_learning_rate,
                                                                      power=power,
                                                                      cycle=cycle,
                                                                      name=name)

@LR_SCHEDULERS.register
class ExponentialDecay(Warmup):
    def __init__(self,
                 initial_learning_rate,
                 train_steps,
                 decay_rate,
                 staircase=False,
                 warmup_learning_rate=0.,
                 warmup_steps=0,
                 name=None):
        super(ExponentialDecay, self).__init__(warmup_steps=warmup_steps,
                                               start_learning_rate=warmup_learning_rate,
                                               end_learning_rate=initial_learning_rate)
        self.decay_fn = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                                                       decay_steps=train_steps - self.warmup_steps,
                                                                       decay_rate=decay_rate,
                                                                       staircase=staircase,
                                                                       name=name)


@LR_SCHEDULERS.register
class CosineDecay(Warmup):
    def __init__(self,
                 initial_learning_rate,
                 train_steps,
                 alpha=0.0,
                 warmup_learning_rate=0.,
                 warmup_steps=0,
                 name=None):
        super(CosineDecay, self).__init__(warmup_steps=warmup_steps,
                                          start_learning_rate=warmup_learning_rate,
                                          end_learning_rate=initial_learning_rate)
        self.decay_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=initial_learning_rate,
                                                          decay_steps=train_steps - self.warmup_steps,
                                                          alpha=alpha,
                                                          name=name)


@LR_SCHEDULERS.register
class LinearCosineDecay(Warmup):
    def __init__(self,
                 initial_learning_rate,
                 train_steps,
                 num_periods=0.5,
                 alpha=0.0,
                 beta=0.001,
                 warmup_learning_rate=0.,
                 warmup_steps=0,
                 name=None):
        super(LinearCosineDecay, self).__init__(warmup_steps=warmup_steps,
                                                start_learning_rate=warmup_learning_rate,
                                                end_learning_rate=initial_learning_rate)
        self.decay_fn = tf.keras.experimental.LinearCosineDecay(initial_learning_rate=initial_learning_rate,
                                                                decay_steps=train_steps - self.warmup_steps,
                                                                num_periods=num_periods,
                                                                alpha=alpha,
                                                                beta=beta,
                                                                name=name)
