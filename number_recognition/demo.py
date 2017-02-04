from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import sys
import random
from PIL import Image, ImageOps

# Load own image and prepare for recognition
im = Image.open("1.png")
im = im.rotate(270).transpose(Image.FLIP_LEFT_RIGHT).convert('L')
im = ImageOps.invert(im)
image = im.load()

label = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] # Declare as digit 1

# Load image into vector with 28 * 28 items (TF needs)
img = []
for x in range(0, im.size[0]):
    for y in range(0, im.size[1]):
        img.append(image[x, y]*(1/255))

        # img.append(image[x, y]*(1/255))

# Pick random image for validation
# val = random.randint(0, 10000)
# img = mnist.test.images[val]
# label = mnist.test.labels[val]

# Define placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define activation function
y = tf.nn.softmax(tf.matmul(x, W) + b)
# y = tf.nn.sigmoid(tf.matmul(x, W) + b)

# Define function for backpropagation
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train model
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Print accurancy
print(sess.run(accuracy, feed_dict={x: [img], y_: [label]}))

# Print image to terminal
for x in range(0, len(img)):
    if img[x] > 0.1:
        sys.stdout.write('*')
    else:
        sys.stdout.write(' ')
    if x % 28 == 0:
        sys.stdout.write("\n")
sys.stdout.write("\n")
print(label)

# Print accurance for whole validation set
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
