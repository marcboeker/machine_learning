import numpy
import pandas

# This is a simple neuronal network with non-linearity using the sigmoid
# function. It has 3 layers in total. Input + output + 1 hidden layer.
class NeuronalNetwork:
    def __init__(self, epochs, learnrate, x, y):
        numpy.random.seed(1)

        self.epochs = epochs
        self.learnrate = learnrate

        self.x = x
        self.y = y

        x_shape = tuple(reversed(self.x.shape))
        y_shape = self.y.shape

        self.weight_0 = 2 * numpy.random.random(x_shape) - 1
        self.weight_1 = 2 * numpy.random.random(y_shape) - 1

    def sigmoid(self, x, derive = False):
        if derive:
            return x * (1 - x)
        return 1 / (1 + numpy.exp(-x))

    def train(self):
        for i in range(0, self.epochs):
          # Feed forward
          layer_0, layer_1, layer_2 = self.run(self.x)

          # Backpropagation
          layer_2_error = self.y - layer_2

          if i % 1000 == 0:
            print("Training error for step %s: %s" % (i, numpy.mean(numpy.abs(layer_2_error))))

          layer_2_delta = layer_2_error * self.sigmoid(layer_2, True)

          layer_1_error = layer_2_delta.dot(self.weight_1.T)
          layer_1_delta = layer_1_error * self.sigmoid(layer_1, True)

          self.weight_0 += self.learnrate * layer_0.T.dot(layer_1_delta)
          self.weight_1 += self.learnrate * layer_1.T.dot(layer_2_delta)

    def run(self, x):
      layer_0 = x
      layer_1 = self.sigmoid(numpy.dot(layer_0, self.weight_0))
      layer_2 = self.sigmoid(numpy.dot(layer_1, self.weight_1))

      return (layer_0, layer_1, layer_2)

    def validate(self, x):
        prediction = nn.run(numpy.array(x))
        print("Your prediction for %s is: %s" % (x, prediction[-1]))

# Trainingdata consists of vectors, where 1 = row contains a value > 0 and 0 = row is 0
x = numpy.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 0, 0]])

# Labels mean 1 = all rows in vector are > 0, 0 = one or many rows in vector are 0
y = numpy.array([[0], [0], [0], [0], [0], [0], [0], [1], [1]])

nn = NeuronalNetwork(60000, 0.01, x, y)
nn.train()
nn.validate([[4, 4, 4]]) # 0.999XX means, that the network is pretty sure, that [4, 4, 4] is a vector, where all rows have values > 0
