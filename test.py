import os
import glob
import argparse
import tensorflow as tf 
from sklearn.metrics import confusion_matrix


def parse(serialized, input_size):
    key_to_features = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string, ""),
        "image/height": tf.io.FixedLenFeature([], tf.int64, 0),
        "image/width": tf.io.FixedLenFeature([], tf.int64, 0),
        "image/channels": tf.io.FixedLenFeature([], tf.int64, 0),
        "image/label": tf.io.FixedLenFeature([], tf.int64, 0)
    }
    parsed_features = tf.io.parse_single_example(serialized, features=key_to_features)

    img_height = parsed_features["image/height"]
    img_width = parsed_features["image/width"]
    # channels = parsed_features["image/channels"]

    image = tf.io.decode_image(parsed_features["image/encoded"], channels=3)
    image = tf.reshape(image, [img_height, img_width, 3])
    
    image = tf.image.resize(image, input_size)
    image = tf.cast(image, tf.uint8)
    labels = tf.cast(parsed_features["image/label"], tf.int64)

    return image, labels


def get_dataset(data_dir, batch_size=32, input_size=(224, 224)):
    tfrecord_path = glob.glob(os.path.join(data_dir, "*.tfrec"))

    if len(tfrecord_path) <= 0:
        raise ValueError("Cannot find tfrecord in %s" % data_dir)

    with tf.device("/cpu:0"):
        ds = tf.data.TFRecordDataset(tfrecord_path)
        ds = ds.map(map_func=lambda x: parse(x, input_size))

        ds = ds.batch(batch_size=batch_size, drop_remainder=False)

        return ds.prefetch(tf.data.experimental.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--saved_model_dir", type=str, required=True, help="The saved model dir.")
    parser.add_argument("--data_dir", type=str, required=True, help="The directory containing *.tfrec.")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    loaded = tf.saved_model.load(args.saved_model_dir, tags=[tf.saved_model.SERVING])
    infer = loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    
    batch_size = args.batch_size
    input_size = (args.input_size, args.input_size)
    gt_labels_list = []
    pred_labels_list = []
    acc_metric = tf.keras.metrics.Accuracy()
    cnt = 0
    for image_batch, label_batch in get_dataset(args.data_dir, batch_size, input_size):
        preds = infer(image_batch)

        acc_metric.update_state(y_true=label_batch, y_pred=preds["labels"])

        gt_labels = label_batch.numpy().tolist()
        pred_labels = preds["labels"].numpy().tolist()
        gt_labels_list.extend(gt_labels)
        pred_labels_list.extend(pred_labels)

    matrix = confusion_matrix(gt_labels_list, pred_labels_list)
    print(matrix)
    print("Accuracy: ", acc_metric.result().numpy())
    

if __name__ == "__main__":
    main()
    