import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter

# Hyperparams
BATCH_SIZE = 5
LEARNING_RATE = 1.2
SKIP_POPULAR_WORDS = 0.22
SKIP_UNPOPULAR_WORDS = 80

# Some parts of the pre-processing code are taken from Andrew Trask
g = open('reviews.txt', 'r')
reviews = list(map(lambda x: x[:-1], g.readlines()))
g.close()

g = open('labels.txt', 'r')
labels = list(map(lambda x: x[:-1].lower(), g.readlines()))
g.close()

# Build list of all words and count their occurance
words = Counter()
for i in range(0, len(reviews)):
    for word in reviews[i].split(' '):
        words[word] += 1

# Pick the top words, exclude the highes and lowest ranked words
total_words = len(words)
upper = int((total_words / 100.0) * SKIP_POPULAR_WORDS)
lower = int((total_words / 100.0) * SKIP_UNPOPULAR_WORDS)

skipping_words = list(map(lambda x: x[0], list(words.most_common())[:upper]))
print("Skipping words: %s" % skipping_words)

word_list = list(map(lambda x: x[0], list(words.most_common())[upper:-lower]))
word2index = {}
for i in range(0, len(word_list)):
    word2index[word_list[i]] = i

# Iterate all reviews and convert to vector with each word as index
dim = len(word_list)
features = np.zeros((len(reviews), dim), dtype=np.float32)
targets = np.zeros((len(reviews), 1), dtype=np.float32)
for i in range(0, len(reviews)):
    review = reviews[i]
    label = labels[i]

    for word in review.split(' '):
        if word in word2index:
            features[i][word2index[word]] = 1

    if label == 'positive':
        targets[i] = 1.0
    else:
        targets[i] = 0.0

reviews = None
labels = None

# Split between training and test data
training_features = features[:-1000]
training_targets = targets[:-1000]
test_features = features[24000:]
test_targets = targets[24000:]

# Define placeholders
x = tf.placeholder(tf.float32, [None, dim])
y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([dim, 1]))
b = tf.Variable(tf.zeros([1]))

# Define activation function
model = tf.nn.sigmoid(tf.matmul(x, W))

# Define function for backpropagation
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(model, y)
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train model
    for i in range(0, 24000):
        # Batch params
        batch = np.random.choice(training_features.shape[0], size=BATCH_SIZE)
        batch = np.array(batch)
        for f, t in zip(training_features[batch], training_targets[batch]):
            batch_xs = np.reshape(f, (1, dim))
            batch_ys = np.reshape(t, (1, 1))
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        if i % 10 == 0:
            print("Trained so far: %s" % i)

    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    fig, ax = plt.subplots(figsize=(8,4))

    # Predict
    correct = 0
    incorrect = 0
    predictions = []
    for feature, target in zip(test_features, test_targets):
        input = np.reshape(feature, (1, dim))
        expected = np.reshape(target, (1, 1))
        prediction = sess.run(model, feed_dict={x: input})

        predictions.append(prediction[0])
        if prediction[0][0] > 0.5 and expected[0][0] == 1.0:
            correct += 1
        elif prediction[0][0] <= 0.5 and expected[0][0] == 0.0:
            correct += 1
        else:
            incorrect += 1

    total = len(test_features)
    print("Total: %s, Correct: %s, Incorrect: %s, Accurancy: %s" % (total, correct, incorrect, (float(correct) / float(total))))

    # Show chart with result
    ax.bar(0.5, total, 1, label='Total')
    ax.bar(1.5, correct, 1, label='Predicted correct')
    ax.bar(2.5, incorrect, 1, label='Predicted incorrect')
    ax.legend()
    plt.show()
