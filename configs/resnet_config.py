import math
from configs import Config


def default_configs(name, batch_size=4, image_size=512):
    h = Config()

    h.dtype = "float32"

    # backbone
    h.model = dict(model=name,
                   convolution="conv2d",
                   dropblock=None,
                   #   dropblock=dict(keep_prob=None,
                   #                  block_size=None)
                   normalization=dict(normalization="batch_norm",
                                      momentum=0.99,
                                      epsilon=1e-3,
                                      axis=-1,
                                      trainable=True),
                   activation=dict(activation="relu"),
                   strides=[2, 1, 2, 2, 2, 1, 2, 1],
                   dilation_rates=[1, 1, 1, 1, 1, 1, 1, 1],
                   output_indices=[-1, ],
                   frozen_stages=[-1, ],
                   num_classes=1)
    
    # loss
    h.use_sigmoid = True
    h.loss=dict(loss="BinaryCrossEntropy", weight=1., from_logits=True, reduction="none")
    h.weight_decay = 4e-5

    # dataset
    h.num_classes = 1
    h.train=dict(dataset=dict(dataset="SmokingDataset",
                              batch_size=batch_size,
                              dataset_dir="/data/bail/smoking/train",
                              training=True,
                              augmentations=[
                                  dict(Resize=dict(input_size=image_size)),
                                #   dict(RandAugment=dict(num_layers=2, 
                                #                         magnitude=10., 
                                #                         cutout_const=40., 
                                #                         translate_const=100.))
                              ]))
    h.val=dict(dataset=dict(dataset="SmokingDataset", 
                            batch_size=batch_size,  
                            dataset_dir="/data/bail/smoking/val", 
                            training=False, 
                            augmentations=[
                                dict(Resize=dict(input_size=image_size))
                            ]))
  
    # train
    h.pretrained_weights_path = "/data/bail/pretrained_weights/efficientnet-b%d.h5" % phi

    h.optimizer = dict(optimizer="SGD", momentum=0.9)
    h.lookahead = None

    h.learning_rate_scheduler = dict(scheduler="CosineDecay", 
                                     initial_learning_rate=0.016,
                                     warmup_steps=800,
                                     warmup_learning_rate=0.001,
                                     train_steps=40001)
    h.checkpoint_dir = "checkpoints/%s" % name
    h.summary_dir = "logs/%s" % name

    h.gradient_clip_norm = 0.
    h.log_every_n_steps = 100
    h.save_ckpt_steps = 2000
    h.val_every_n_steps = 2000

    return h


efficientdet_model_param_dict = {
    "EfficientNetB0": dict(phi=0, batch_size=32, image_size=224),
    "EfficientNetB1": dict(phi=1, batch_size=32, image_size=240),
    "EfficientNetB2": dict(phi=2, batch_size=4, image_size=260),
    "EfficientNetB3": dict(phi=3, batch_size=4, image_size=300),
    "EfficientNetB4": dict(phi=4, batch_size=4, image_size=380),
    "EfficientNetB5": dict(phi=5, batch_size=4, image_size=456),
    "EfficientNetB6": dict(phi=6, batch_size=4, image_size=528),
    "EfficientNetB7": dict(phi=7, batch_size=4, image_size=600),
}


def get_efficientnet_config(model_name="EfficientNetB0"):
    return default_configs(**efficientdet_model_param_dict[model_name])


if __name__ == "__main__":
    print(get_efficientdet_config("EfficientNetB0"))
