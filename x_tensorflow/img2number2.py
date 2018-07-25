import tensorflow as tf
import cv2
import numpy as np

print(tf.__version__)

images = []

image = cv2.imread('field_7.png', cv2.IMREAD_GRAYSCALE)
image = np.array(image, dtype=np.uint8)
image = image.astype('float32')
images.append(image)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model/model.ckpt-50000.meta')
    saver.restore(sess, 'model/model.ckpt-50000')
    graph = tf.get_default_graph()
    y = tf.nn.softmax(image)
    x = tf.placeholder(tf.float32, shape=[1, 28, 28], name='x')
    a = sess.run(y, feed_dict={x: images})
    print(np.argmax(y[0]))
