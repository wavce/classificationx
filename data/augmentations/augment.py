import math
import numpy as np
import tensorflow as tf


IMAGE_SIZE = 224
CROP_PADDING = 32


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
    """Generates cropped_image using one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
        image: `Tensor` of binary image data.
        bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
            where each coordinate is [0, 1) and the coordinates are arranged
            as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding
            box supplied.
        aspect_ratio_range: An optional list of `float`s. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `float`s. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
    Returns:
        cropped image `Tensor`
    """
    with tf.name_scope('distorted_bounding_box_crop'):
        shape = tf.shape(image)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        
        image = tf.image.crop_to_bounding_box(
            image,
            offset_height=offset_y,
            offset_width=offset_x,
            target_height=target_height,
            target_width=target_width)

        return image


def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def _resize_image(image, image_size, method=None):
    if method is not None:
        return tf.image.resize([image], [image_size, image_size], method)[0]

    return tf.image.resize([image], [image_size, image_size], method="bicubic")[0]


def _center_crop(image, image_size, resize_method=None):
    """Crops to center of image with padding then scales image_size."""
    with tf.name_scope("center_crop"):
        shape = tf.shape(image)
        image_height = shape[0]
        image_width = shape[1]

        padded_center_crop_size = tf.cast(
            ((image_size / (image_size + CROP_PADDING)) *
            tf.cast(tf.minimum(image_height, image_width), tf.float32)),
            tf.int32)

        offset_height = ((image_height - padded_center_crop_size) + 1) // 2
        offset_width = ((image_width - padded_center_crop_size) + 1) // 2

        image = tf.image.crop_to_bounding_box(
            image, 
            offset_height=offset_height, 
            offset_width=offset_width,
            target_height=padded_center_crop_size,
            target_width=padded_center_crop_size)
        image = _resize_image(image, image_size, resize_method)

        return image


def _random_distorted_crop(image, image_size, resize_method=None):
    """Make a random crop of image_size."""
    with tf.name_scope("random_distorted_crop"):
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        original_shape = tf.shape(image)
        image = distorted_bounding_box_crop(
            image,
            bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(3. / 4, 4. / 3.),
            area_range=(0.5, 1.0),
            max_attempts=10)
        bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

        image = tf.cond(
            bad,
            lambda: _center_crop(image, image_size),
            lambda: _resize_image(image, image_size, resize_method))

        return image


class RandomDistortedCrop(object):
    def __init__(self, input_size, resize_method=None):
        self.input_size = input_size
        self.resize_method = resize_method

    def __call__(self, image):
        return _random_distorted_crop(image, self.input_size, self.resize_method)


class CenterCrop(object):
    def __init__(self, input_size, resize_method=None):
        self.input_size = input_size
        self.resize_method = resize_method
    
    def __call__(self, image):
        return _center_crop(image, self.input_size, self.resize_method)


class Resize(object):
    def __init__(self, input_size, resize_method=None):
        self.input_size = input_size
        self.resize_method = resize_method
    
    def __call__(self, image):
        return _resize_image(image, self.input_size, self.resize_method)


class RandomDistortColor(object):
    def __init__(self,
                 brightness=32./255.,
                 min_saturation=0.5,
                 max_saturation=1.5,
                 hue=0.2,
                 min_contrast=0.5,
                 max_contrast=1.5,
                 **Kwargs):
        self.brightness = brightness
        self.min_saturation = min_saturation
        self.max_saturation = max_saturation
        self.hue = hue
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def _distort_color0(self, image):
        image = tf.image.random_brightness(image, max_delta=self.brightness)
        image = tf.image.random_saturation(image, lower=self.min_saturation, upper=self.max_saturation)
        image = tf.image.random_hue(image, max_delta=self.hue)
        image = tf.image.random_contrast(image, lower=self.min_contrast, upper=self.max_contrast)

        return image

    def _distort_color1(self, image):
        image = tf.image.random_saturation(image, lower=self.min_saturation, upper=self.max_saturation)
        image = tf.image.random_brightness(image, max_delta=self.brightness)
        image = tf.image.random_contrast(image, lower=self.min_contrast, upper=self.max_contrast)
        image = tf.image.random_hue(image, max_delta=self.hue)

        return image
    
    def _distort_color2(self, image):
        image = tf.image.random_contrast(image, lower=self.min_contrast, upper=self.max_contrast)
        image = tf.image.random_hue(image, max_delta=self.hue)
        image = tf.image.random_brightness(image, max_delta=self.brightness)
        image = tf.image.random_saturation(image, lower=self.min_saturation, upper=self.max_saturation)

        return image
    
    def _distort_color3(self, image):
        image = tf.image.random_hue(image, max_delta=self.hue)
        image = tf.image.random_saturation(image, lower=self.min_saturation, upper=self.max_saturation)
        image = tf.image.random_contrast(image, lower=self.min_contrast, upper=self.max_contrast)
        image = tf.image.random_brightness(image, max_delta=self.brightness)

        return image

    def __call__(self, image):
        with tf.name_scope("distort_color"):
            rand_int = tf.random.uniform([], 0, 4, tf.int32)
            if rand_int == 0:
                image = self._distort_color0(image)
            elif rand_int == 1:
                image = self._distort_color1(image)
            elif rand_int == 2:
                image = self._distort_color2(image)
            else:
                image = self._distort_color3(image)
            
            image = tf.minimum(tf.maximum(image, 0), 255)

            return image
