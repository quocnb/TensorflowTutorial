from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Using Logistic Regression (Softmax)
X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

W = tf.Variable(initial_value=tf.zeros([784, 10]))
h = tf.Variable(initial_value=tf.zeros([10]))

y_pred = tf.nn.softmax(tf.matmul(X, W) + h)

cross_sum = tf.reduce_sum(y * tf.log(y_pred), axis=1)
cross_entropy = tf.reduce_mean(-cross_sum)
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys})

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})
print(test_accuracy)
sess.close()
