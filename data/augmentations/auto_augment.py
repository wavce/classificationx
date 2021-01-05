# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AutoAugment and RandAugment policies for enhanced image preprocessing.
AutoAugment Reference: https://arxiv.org/abs/1805.09501
RandAugment Reference: https://arxiv.org/abs/1909.13719
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow_addons import image as image_ops

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.


def rotate(image, degrees, replace):
    """Rotates the image by degrees either clockwise or counterclockwise.
    Args:
        image: An image Tensor of type uint8.
        degrees: Float, a scalar angle in degrees to rotate all images by. If
        degrees is positive the image will be rotated clockwise otherwise it will
        be rotated counterclockwise.
        replace: A one or three value 1D tensor to fill empty pixels caused by
        the rotate operation.
    Returns:
        The rotated version of image.
    """
    with tf.name_scope("rotate"):
        # Convert from degrees to radians.
        degrees_to_radians = math.pi / 180.0
        radians = degrees * degrees_to_radians

        # In practice, we should randomize the rotation degrees by flipping
        # it negatively half the time, but that's done on 'degrees' outside
        # of the function.
        image = image_ops.rotate(wrap(image), radians)
        return unwrap(image, replace)


def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.
    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.
    Args:
        image1: An image Tensor of type uint8.
        image2: An image Tensor of type uint8.
        factor: A floating point value above 0.0.
    Returns:
        A blended image Tensor of type uint8.
    """
    with tf.name_scope("blend"):
        if factor == 0.0:
            return tf.convert_to_tensor(image1)
        if factor == 1.0:
            return tf.convert_to_tensor(image2)

        image1 = tf.cast(image1, tf.float32)
        image2 = tf.cast(image2, tf.float32)

        difference = image2 - image1
        scaled = factor * difference

        # Do addition in float.
        temp = tf.cast(image1, tf.float32) + scaled

        # Interpolate
        if factor > 0.0 and factor < 1.0:
            # Interpolation means we always stay within 0 and 255.
            return tf.cast(temp, tf.uint8)

        # Extrapolate:
        #
        # We need to clip and then cast.
        return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


def cutout(image, pad_size, replace=0):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.
    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.
    
    Args:
        image: An image Tensor of type uint8.
        pad_size: Specifies how big the zero mask that will be generated is that
        is applied to the image. The mask will be of size
        (2*pad_size x 2*pad_size).
        replace: What pixel value to fill in the image in the area that has
        the cutout mask applied to it.
    
    Returns:
        An image Tensor that is of type uint8.
    """
    with tf.name_scope("cutout"):
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        # Sample the center location in the image where the zero mask will be applied.
        cutout_center_height = tf.random.uniform(
            shape=[], minval=0, maxval=image_height,
            dtype=tf.int32)

        cutout_center_width = tf.random.uniform(
            shape=[], minval=0, maxval=image_width,
            dtype=tf.int32)

        lower_pad = tf.maximum(0, cutout_center_height - pad_size)
        upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
        left_pad = tf.maximum(0, cutout_center_width - pad_size)
        right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

        cutout_shape = [image_height - (lower_pad + upper_pad),
                        image_width - (left_pad + right_pad)]
        padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
        mask = tf.pad(tf.zeros(cutout_shape, dtype=image.dtype),
                    padding_dims, constant_values=1)
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, 3])
        image = tf.where(tf.equal(mask, 0),
                        tf.ones_like(image, dtype=image.dtype) * replace,
                        image)
        
        return image


def solarize(image, threshold=128):
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return tf.where(image < threshold, image, 255 - image, name="solarize")


def solarize_add(image, addition=0, threshold=128):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    with tf.name_scope("solarize_add"):
        added_image = tf.cast(image, tf.int64) + addition
        added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
        return tf.where(image < threshold, added_image, image)


def color(image, factor):
    """Equivalent of PIL Color."""
    with tf.name_scope("color"):
        degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
        return blend(degenerate, image, factor)


def contrast(image, factor):
    """Equivalent of PIL Contrast."""
    with tf.name_scope("contrast"):
        degenerate = tf.image.rgb_to_grayscale(image)
        # Cast before calling tf.histogram.
        degenerate = tf.cast(degenerate, tf.int32)

        # Compute the grayscale histogram, then compute the mean pixel value,
        # and create a constant image size of that value.  Use that as the
        # blending degenerate target of the original image.
        hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
        mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
        degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
        degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))

        return blend(degenerate, image, factor)


def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    with tf.name_scope("brightness"):
        degenerate = tf.zeros_like(image)
        return blend(degenerate, image, factor)


def posterize(image, bits):
    """Equivalent of PIL Posterize."""
    with tf.name_scope("posterize"):
        shift = 8 - bits
        return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def translate_x(image, pixels, replace):
    """Equivalent of PIL Translate in X dimension."""
    with tf.name_scope("translate_x"):
        image = image_ops.translate(wrap(image), [-pixels, 0])
        return unwrap(image, replace)


def translate_y(image, pixels, replace):
    """Equivalent of PIL Translate in Y dimension."""
    with tf.name_scope("translate_y"):
        image = image_ops.translate(wrap(image), [0, -pixels])
        return unwrap(image, replace)


def shear_x(image, level, replace):
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    with tf.name_scope("shear_x"):
        image = image_ops.transform(images=wrap(image), 
                                    transforms=[1., level, 0., 0., 1., 0., 0., 0.])
        return unwrap(image, replace)


def shear_y(image, level, replace):
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
    with tf.name_scope("shear_y"):
        image = image_ops.transform(images=wrap(image), 
                                    transforms=[1., 0., 0., level, 1., 0., 0., 0.])
        return unwrap(image, replace)


def autocontrast(image):
    """Implements Autocontrast function from PIL using TF ops.
    Args:
        image: A 3D uint8 tensor.
    Returns:
        The image after it has had autocontrast applied to it and will be of type
        uint8.
    """

    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), tf.float32)
        hi = tf.cast(tf.reduce_max(image), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    with tf.name_scope("autocontrast"):
        s1 = scale_channel(image[:, :, 0])
        s2 = scale_channel(image[:, :, 1])
        s3 = scale_channel(image[:, :, 2])
        image = tf.stack([s1, s2, s3], 2)

        return image


def sharpness(image, factor):
    """Implements Sharpness function from PIL using TF ops."""
    with tf.name_scope("sharpness"):
        orig_image = image
        image = tf.cast(image, tf.float32)
        # Make image 4D for conv operation.
        image = tf.expand_dims(image, 0)
        # SMOOTH PIL Kernel.
        kernel = tf.constant(
            [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
            shape=[3, 3, 1, 1]) / 13.
        # Tile across channel dimension.
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        strides = [1, 1, 1, 1]
        with tf.device('/cpu:0'):
            degenerate = tf.nn.depthwise_conv2d(image, kernel, strides, padding='VALID')
        degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

        # For the borders of the resulting image, fill in the values of the
        # original image.
        mask = tf.ones_like(degenerate)
        padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
        padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
        result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

        # Blend the final result.
        return blend(result, orig_image, factor)


def equalize(image):
    """Implements Equalize function from PIL using TF ops."""
    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0),
                         lambda: im,
                         lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    with tf.name_scope("equalize"):
        s1 = scale_channel(image, 0)
        s2 = scale_channel(image, 1)
        s3 = scale_channel(image, 2)
        image = tf.stack([s1, s2, s3], 2)

        return image


def invert(image):
    """Inverts the image pixels."""
    with tf.name_scope("invert"):
        image = tf.convert_to_tensor(image)
        return 255 - image


def wrap(image):
    """Returns 'image' with an extra channel set to all 1s."""
    with tf.name_scope("wrap"):
        shape = tf.shape(image)
        extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
        extended = tf.concat([image, extended_channel], axis=2)
        
        return extended


def unwrap(image, replace):
    """Unwraps an image produced by wrap.
    Where there is a 0 in the last channel for every spatial position,
    the rest of the three channels in that spatial dimension are grayed
    (set to 128).  Operations like translate and shear on a wrapped
    Tensor will leave 0s in empty locations.  Some transformations look
    at the intensity of values to do preprocessing, and we want these
    empty pixels to assume the 'average' value, rather than pure black.
    Args:
        image: A 3D Image Tensor with 4 channels.
        replace: A one or three value 1D tensor to fill empty pixels.
    Returns:
        image: A 3D image Tensor with 3 channels.
    """
    with tf.name_scope("unwrap"):
        image_shape = tf.shape(image)
        # Flatten the spatial dimensions.
        flattened_image = tf.reshape(image, [-1, image_shape[2]])

        # Find all pixels where the last channel is zero.
        alpha_channel = tf.expand_dims(flattened_image[:, 3], -1)

        replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

        # Where they are zero, fill them in with 'replace'.
        flattened_image = tf.where(
            tf.equal(alpha_channel, 0),
            tf.ones_like(flattened_image, dtype=image.dtype) * replace,
            flattened_image)

        image = tf.reshape(flattened_image, image_shape)
        image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
        
        return image


def _randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    with tf.name_scope("randomly_negate_tensor"):
        should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
        final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
        return final_tensor


def _rotate_level_to_arg(level):
    with tf.name_scope("rotate_level_to_arg"):
        level = (level/_MAX_LEVEL) * 30.
        level = _randomly_negate_tensor(level)
        return (level, )


def _shrink_level_to_arg(level):
    """Converts level to ratio by which we shrink the image content."""
    if level == 0:
        return (1.0, )  # if level is zero, do not shrink the image
    # Maximum shrinking ratio is 2.9.
    level = 2. / (_MAX_LEVEL / level) + 0.9
    return (level, )


def _enhance_level_to_arg(level):
    return ((level/_MAX_LEVEL) * 1.8 + 0.1, )


def _shear_level_to_arg(level):
    with tf.name_scope("shear_level_to_arg"):
        level = (level/_MAX_LEVEL) * 0.3
        # Flip level to negative with 50% chance.
        level = _randomly_negate_tensor(level)
        
        return (level, )


def _translate_level_to_arg(level, translate_const):
    with tf.name_scope("translate_level_to_arg"):
        level = (level/_MAX_LEVEL) * float(translate_const)
        # Flip level to negative with 50% chance.
        level = _randomly_negate_tensor(level)
        return (level, )


def _mult_to_arg(level: float, multiplier: float = 1.):
    return (int((level / _MAX_LEVEL) * multiplier),)


def _apply_func_with_prob(func, image, args, prob):
    """Apply `func` to image w/ `args` as input with probability `prob`."""
    assert isinstance(args, tuple)

    with tf.name_scope("apply_func_with_prob"):
        # Apply the function with probability `prob`.
        should_apply_op = tf.cast(
            tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
        augmented_image = tf.cond(should_apply_op,
                                  lambda: func(image, *args),
                                  lambda: image)
        return augmented_image


def select_and_apply_random_policy(policies, image):
    """Select a random policy from `policies` and apply it to `image`."""
    with tf.name_scope("select_and_apply_random_policy"):
        policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
        # Note that using tf.case instead of tf.conds would result in significantly
        # larger graphs and would even break export for some larger policies.
        for (i, policy) in enumerate(policies):
            image = tf.cond(tf.equal(i, policy_to_select),
                            lambda selected_policy=policy: selected_policy(image),
                            lambda: image)
        return image


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
}

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset({
    'Rotate',
    'TranslateX',
    'ShearX',
    'ShearY',
    'TranslateY',
    'Cutout',
})


def level_to_arg(cutout_const, translate_const):
    """Creates a dict mapping image operation names to their arguments."""

    no_arg = lambda level: ()
    posterize_arg = lambda level: _mult_to_arg(level, 4)
    solarize_arg = lambda level: _mult_to_arg(level, 256)
    solarize_add_arg = lambda level: _mult_to_arg(level, 110)
    cutout_arg = lambda level: _mult_to_arg(level, cutout_const)
    translate_arg = lambda level: _translate_level_to_arg(level, translate_const)

    args = {
        'AutoContrast': no_arg,
        'Equalize': no_arg,
        'Invert': no_arg,
        'Rotate': _rotate_level_to_arg,
        'Posterize': posterize_arg,
        'Solarize': solarize_arg,
        'SolarizeAdd': solarize_add_arg,
        'Color': _enhance_level_to_arg,
        'Contrast': _enhance_level_to_arg,
        'Brightness': _enhance_level_to_arg,
        'Sharpness': _enhance_level_to_arg,
        'ShearX': _shear_level_to_arg,
        'ShearY': _shear_level_to_arg,
        'Cutout': cutout_arg,
        'TranslateX': translate_arg,
        'TranslateY': translate_arg,
    }
    return args


def _parse_policy_info(name, prob, level, replace_value, cutout_const, translate_const):
    """Return the function that corresponds to `name` and update `level` param."""
    func = NAME_TO_FUNC[name]
    args = level_to_arg(cutout_const, translate_const)[name](level)

    if name in REPLACE_FUNCS:
        # Add in replace arg if it is required for the function that is called.
        args = tuple(list(args) + [replace_value])

    return func, prob, args


class ImageAugment(object):
  """Image augmentation class for applying image distortions."""

  def __call__(self, image):
    """Given an image tensor, returns a distorted image with the same shape.
    
    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.
    
    Returns:
      The augmented version of `image`.
    """
    raise NotImplementedError()


class AutoAugment(ImageAugment):
    """Applies the AutoAugment policy to images.
       AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.
    """

    def __init__(self,
                augmentation_name='v0',
                policies=None,
                cutout_const=100,
                translate_const=250):
        """Applies the AutoAugment policy to images.
        Args:
            augmentation_name: The name of the AutoAugment policy to use. The
                available options are `v0` and `test`. `v0` is the policy used for all
                of the results in the paper and was found to achieve the best results on
                the COCO dataset. `v1`, `v2` and `v3` are additional good policies found
                on the COCO dataset that have slight variation in what operations were
                used during the search procedure along with how many operations are
                applied in parallel to a single image (2 vs 3).
            policies: list of lists of tuples in the form `(func, prob, level)`,
                `func` is a string name of the augmentation function, `prob` is the
                probability of applying the `func` operation, `level` is the input
                argument for `func`.
            cutout_const: multiplier for applying cutout.
            translate_const: multiplier for applying translation.
        """
        super(AutoAugment, self).__init__()

        if policies is None:
            self.available_policies = {
                'v0': self.policy_v0(),
                'test': self.policy_test(),
                'simple': self.policy_simple(),
            }

        if augmentation_name not in self.available_policies:
            raise ValueError(
                'Invalid augmentation_name: {}'.format(augmentation_name))

        self.augmentation_name = augmentation_name
        self.policies = self.available_policies[augmentation_name]
        self.cutout_const = float(cutout_const)
        self.translate_const = float(translate_const)

    def __call__(self, image):
        """Applies the AutoAugment policy to `image`.
        AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.
        
        Args:
            image: `Tensor` of shape [height, width, 3] representing an image.
        
        Returns:
            A version of image that now has data augmentation applied to it based on
            the `policies` pass into the function.
        """
        input_image_type = image.dtype

        if input_image_type != tf.uint8:
            image = tf.clip_by_value(image, 0.0, 255.0)
            image = tf.cast(image, dtype=tf.uint8)

        replace_value = [128] * 3

        # func is the string name of the augmentation function, prob is the
        # probability of applying the operation and level is the parameter
        # associated with the tf op.

        # tf_policies are functions that take in an image and return an augmented
        # image.
        tf_policies = []
        for policy in self.policies:
            tf_policy = []
            # Link string name to the correct python function and make sure the
            # correct argument is passed into that function.
            for policy_info in policy:
                policy_info = list(policy_info) + [
                    replace_value, self.cutout_const, self.translate_const
                ]
                tf_policy.append(_parse_policy_info(*policy_info))
            # Now build the tf policy that will apply the augmentation procedue
            # on image.
            def make_final_policy(tf_policy_):

                def final_policy(image_):
                    for func, prob, args in tf_policy_:
                        image_ = _apply_func_with_prob(func, image_, args, prob)
                    return image_

                return final_policy

            tf_policies.append(make_final_policy(tf_policy))

        image = select_and_apply_random_policy(tf_policies, image)
        image = tf.cast(image, dtype=input_image_type)

        return image

    @staticmethod
    def policy_v0():
        """Autoaugment policy that was used in AutoAugment Paper.
        Each tuple is an augmentation operation of the form
        (operation, probability, magnitude). Each element in policy is a
        sub-policy that will be applied sequentially on the image.
        
        Returns:
            the policy.
        """

        # TODO(dankondratyuk): tensorflow_addons defines custom ops, which
        # for some reason are not included when building/linking
        # This results in the error, "Op type not registered
        # 'Addons>ImageProjectiveTransformV2' in binary" when running on borg TPUs
        policy = [
            [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
            [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
            [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
            [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
            [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
            [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
            [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
            [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
            [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
            [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
            [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
            [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
            [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
            [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
            [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
            [('Rotate', 1.0, 7), ('TranslateY', 0.8, 9)],
            [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
            [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
            [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
            [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
            [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
            [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
            [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
            [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
            [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
        ]
        return policy

    @staticmethod
    def policy_simple():
        """Same as `policy_v0`, except with custom ops removed."""

        policy = [
            [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
            [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
            [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
            [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
            [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
            [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
            [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
            [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
            [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
            [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
            [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
            [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
            [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        ]
        return policy

    @staticmethod
    def policy_test():
        """Autoaugment test policy for debugging."""
        policy = [
            [('TranslateX', 1.0, 4), ('Equalize', 1.0, 10)],
        ]
        return policy


class RandAugment(ImageAugment):
    """Applies the RandAugment policy to images.
    RandAugment is from the paper https://arxiv.org/abs/1909.13719,
    """

    def __init__(self, num_layers=2, magnitude=10., cutout_const=40., translate_const=100.):
        """Applies the RandAugment policy to images.
        
        Args:
            num_layers: Integer, the number of augmentation transformations to apply
                sequentially to an image. Represented as (N) in the paper. Usually best
                values will be in the range [1, 3].
            magnitude: Integer, shared magnitude across all augmentation operations.
                Represented as (M) in the paper. Usually best values are in the range
                [5, 10].
            cutout_const: multiplier for applying cutout.
            translate_const: multiplier for applying translation.
        """
        super(RandAugment, self).__init__()

        self.num_layers = num_layers
        self.magnitude = float(magnitude)
        self.cutout_const = float(cutout_const)
        self.translate_const = float(translate_const)
        self.available_ops = [
            'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize', 'Solarize',
            'Color', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY',
            'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd'
        ]

    def __call__(self, image):
        """Applies the RandAugment policy to `image`.
        Args:
            image: `Tensor` of shape [height, width, 3] representing an image.
        Returns:
            The augmented version of `image`.
        """
        input_image_type = image.dtype

        if input_image_type != tf.uint8:
            image = tf.clip_by_value(image, 0.0, 255.0)
            image = tf.cast(image, dtype=tf.uint8)

        replace_value = [128] * 3
        min_prob, max_prob = 0.2, 0.8

        for _ in range(self.num_layers):
            op_to_select = tf.random.uniform(
                [], maxval=len(self.available_ops) + 1, dtype=tf.int32)

            branch_fns = []
            for (i, op_name) in enumerate(self.available_ops):
                prob = tf.random.uniform([],
                                        minval=min_prob,
                                        maxval=max_prob,
                                        dtype=tf.float32)
                func, _, args = _parse_policy_info(op_name,
                                                prob,
                                                self.magnitude,
                                                replace_value,
                                                self.cutout_const,
                                                self.translate_const)
                branch_fns.append((
                    i,
                    # pylint:disable=g-long-lambda
                    lambda selected_func=func, selected_args=args: selected_func(
                        image, *selected_args)))
                # pylint:enable=g-long-lambda

            image = tf.switch_case(branch_index=op_to_select,
                                   branch_fns=branch_fns,
                                   default=lambda: tf.identity(image))

        image = tf.cast(image, dtype=input_image_type)

        return image


def test():
    import cv2
    import numpy as np

    def parser(serialized):
        key_to_features = {
            'image/width': tf.io.FixedLenFeature([], tf.int64, 0),
            'image/height': tf.io.FixedLenFeature([], tf.int64, 0),
            'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
            'image/object/count': tf.io.FixedLenFeature([], tf.int64, 0),
            'image/object/categories': tf.io.VarLenFeature(tf.int64),
            'image/object/bbox/x': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/y': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/width': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/height': tf.io.VarLenFeature(tf.float32)
        }

        parsed_features = tf.io.parse_single_example(
            serialized, features=key_to_features, name='parse_features')

        image = tf.image.decode_image(parsed_features['image/encoded'])

        image = tf.cond(tf.equal(tf.shape(image)[-1], 1),
                        true_fn=lambda: tf.tile(image, [1, 1, 3]),
                        false_fn=lambda: image)
        img_width = tf.cast(parsed_features['image/width'], tf.int32)
        img_height = tf.cast(parsed_features['image/height'], tf.int32)
        image = tf.reshape(image, [img_height, img_width, 3])
        image = tf.cast(image, tf.uint8)

        labels = tf.cast(tf.sparse.to_dense(parsed_features['image/object/categories']), tf.float32)
        # labels += 1.
        x = tf.cast(tf.sparse.to_dense(parsed_features['image/object/bbox/x']), tf.float32)
        y = tf.cast(tf.sparse.to_dense(parsed_features['image/object/bbox/y']), tf.float32)
        w = tf.cast(tf.sparse.to_dense(parsed_features['image/object/bbox/width']), tf.float32)
        h = tf.cast(tf.sparse.to_dense(parsed_features['image/object/bbox/height']), tf.float32)
    
        # float_img_h = tf.cast(img_height, tf.float32)
        # float_img_w = tf.cast(img_width, tf.float32)

        # x *= float_img_w
        # y *= float_img_h
        # w *= float_img_w
        # h *= float_img_h

        boxes = tf.stack([y, x, y + h, x + w], axis=-1)
        boxes = tf.clip_by_value(boxes, 0, 1)

        image = AutoAugment("v0")(image)

        return image, boxes
    
    
    dataset = tf.data.TFRecordDataset(["/home/bail/Data/data1/Dataset/Objects365/train/train0.tfrecord"])
    dataset = dataset.map(map_func=parser)
    dataset = dataset.batch(batch_size=1, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) 

    for images, boxes in dataset.take(20):
        image = images.numpy()[0]
        boxes = boxes.numpy()[0] * np.array(list(image.shape[:2]) * 2)

        for box in boxes:
            image = cv2.rectangle(image, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (255, 0, 0), 1)
        
        cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)


if __name__ == "__main__":
    test()