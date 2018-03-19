import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
from tensorflow.python.ops import variables
from tensorflow.python.ops import random_ops
from tensorflow.python.framework.ops import convert_to_tensor


if __name__ == '__main__':
    batch_size = 1
    height, width = 224, 224
    with tf.Session() as sess:
        # inputs = random_ops.random_uniform((batch_size, height, width, 3))
        image = tf.image.decode_jpeg(tf.read_file('Aoba.jpg'))
        image_tensor = image.eval()
        image_tensor = tf.expand_dims(image_tensor, 0)
        image_tensor = tf.to_float(image_tensor)

        logits, end_points = vgg.vgg_19(image_tensor)
        sess.run(variables.global_variables_initializer())
        output = sess.run(end_points['vgg_19/pool5'])
        print(output.shape)

