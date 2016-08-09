import click
import numpy as np
import tensorflow as tf


# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])


# We'll want to randomly initialize weights.
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# Our model is a standard 1-hidden-layer multi-layer-perceptron with ReLU
# activation. The softmax (which turns arbitrary real-valued outputs into
# probabilities) gets applied in the cost function.
def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)


def get_actual_fizz_buzz():
    def fizz_buzz(x):
        if   x % 15 == 0: return 'fizzbuzz'
        elif x % 5  == 0: return 'buzz'
        elif x % 3  == 0: return 'fizz'
        else:             return str(x)

    return [fizz_buzz(x) for x in range(1, 101)]


def run(NUM_DIGITS=10, NUM_HIDDEN=100, learning_rate=0.05, iterations=10000):
    # Our goal is to produce fizzbuzz for the numbers 1 to 100. So it would be
    # unfair to include these in our training data. Accordingly, the training data
    # corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).
    trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
    trY = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** NUM_DIGITS)])

    # Our variables. The input has width NUM_DIGITS, and the output has width 4.
    X = tf.placeholder("float", [None, NUM_DIGITS])
    Y = tf.placeholder("float", [None, 4])

    # Initialize the weights.
    w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
    w_o = init_weights([NUM_HIDDEN, 4])

    # Predict y given x using the model.
    py_x = model(X, w_h, w_o)

    # We'll train our model by minimizing a cost function.
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # And we'll make predictions by choosing the largest output.
    predict_op = tf.argmax(py_x, 1)

    # Finally, we need a way to turn a prediction (and an original number)
    # into a fizz buzz output
    def fizz_buzz(i, prediction):
        return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

    BATCH_SIZE = 128

    # Launch the graph in a session
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        train_writer = tf.train.SummaryWriter('/tmp/train', sess.graph)
        merged_summary_op = tf.merge_all_summaries()

        for epoch in range(iterations):
            # Shuffle the data before each training iteration.
            p = np.random.permutation(range(len(trX)))
            trX, trY = trX[p], trY[p]

            # Train in batches of 128 inputs.
            for iteration, start in enumerate(range(0, len(trX), BATCH_SIZE)):
                end = start + BATCH_SIZE
                feed_dict = {
                    X: trX[start:end],
                    Y: trY[start:end]
                }
                summary = sess.run(train_op,
                                   feed_dict)

                if iteration and not iteration % 32:
                    train_writer.add_summary(sess.run(merged_summary_op),
                                             iteration)

            # And print the current accuracy on the training data.
            if not epoch % 250:
                feed_dict = {X: trX, Y: trY}
                print epoch, np.mean(np.argmax(trY, axis=1) ==
                                     sess.run(predict_op, feed_dict))

        # And now for some fizz buzz
        numbers = np.arange(1, 101)
        teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
        teY = sess.run(predict_op, feed_dict={X: teX})
        output = np.vectorize(fizz_buzz)(numbers, teY)

        return output


def performance(tf_output):
    correct = get_actual_fizz_buzz()
    inaccurate = [(pos, val, correct[pos])
                  for (pos, val) in enumerate(tf_output)
                  if val != correct[pos]]

    print len(inaccurate), 'wrong value(s) (position, correct, predicted):'

    for pos, val, correct_val in inaccurate:
        print pos, correct_val, val

    print
    print 'Predictions TensorFlow made:'
    print tf_output


@click.command()
@click.option('--digits', required=False, type=click.INT, default=10)
@click.option('--hidden_units', required=False, type=click.INT, default=100)
@click.option('--learning_rate', required=False, type=click.FLOAT, default=0.05)
@click.option('--iterations', required=False, type=click.INT, default=10000)
def main(digits, hidden_units, learning_rate, iterations):
    tf_output = run(digits, hidden_units, learning_rate, iterations)
    performance(tf_output)


if __name__ == '__main__':
    main()
