import tensorflow as tf

# X = [1, 2, 1, 3]
# y = [1, 0, 1, 0]
# Find W, h for WX+h = y

X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable([1], dtype=tf.float32)
h = tf.Variable([0], dtype=tf.float32)

y_pred = W * X + h

loss = tf.reduce_sum(tf.square(y - y_pred))

gdo = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = gdo.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    sess.run(train, {X: [1, 2, 1, 3], y: [1, 0, 1, 0]})

print(sess.run([W, h]))

new_X = tf.constant([1, 2, 1, 3], dtype=tf.float32)
new_y_pred = W*new_X + h
print('-' * 50)
print(sess.run(new_y_pred))
