import argparse
import tensorflow as tf 
from models import build_model
from configs import build_configs
from core import build_optimizer


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--ckpt_dir", default=None, type=str)
parser.add_argument("--num_classes", default=None, type=int)
parser.add_argument("--use_sigmoid", default=None, type=bool)
parser.add_argument("--saved_model_dir", required=True, type=str)

args = parser.parse_args()


cfg = build_configs(args.model)
if args.num_classes is not None:
    cfg.num_classes = args.num_classes
    cfg.model.num_classes = args.num_classes

if args.ckpt_dir is not None:
    cfg.checkpoint_dir = args.ckpt_dir
    
if args.use_sigmoid is not None:
    cfg.use_sigmoid = args.use_sigmoid
if cfg.num_classes == 1:
    cfg.use_sigmoid = True

base_model = build_model(**cfg.model.as_dict())

x = base_model.output 
print(cfg)
    
if cfg.use_sigmoid:
    x = tf.keras.layers.Activation("sigmoid")(x)
else:
    x = tf.keras.layers.Softmax(axis=-1)(x)

if cfg.num_classes > 1:
    labels = tf.keras.layers.Lambda(lambda inp: tf.argmax(inp, -1), name="labels")(x)
    scores = tf.keras.layers.Lambda(lambda inp: tf.reduce_max(inp, -1), name="scores")(x)
    outputs = [labels, scores]
else:
    scores = tf.keras.layers.Lambda(lambda inp: tf.squeeze(inp, -1), name="scores")(x)
    labels = tf.keras.layers.Lambda(lambda inp: tf.squeeze(tf.cast(inp > 0.5, tf.int64), -1), name="labels")(x)
    outputs = [labels, scores]

model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs)

optimizer = build_optimizer(**cfg.optimizer.as_dict())


checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(
    checkpoint=checkpoint, directory=cfg.checkpoint_dir, max_to_keep=10)

latest_checkpoint = manager.latest_checkpoint
checkpoint.restore(latest_checkpoint)  #.assert_consumed()
    
saved_model_dir = args.saved_model_dir or "./saved_models/" + args.model

tf.saved_model.save(model, saved_model_dir)
print("Saved model to", saved_model_dir)

