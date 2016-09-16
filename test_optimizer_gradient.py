import tensorflow as tf
import numpy as np

def get_name(tensors):
    return "\n".join([t.name for t in tensors])

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
session = tf.Session()

x = tf.placeholder(tf.float32, (None, 2), "x")

with tf.variable_scope("w"):
    w = tf.get_variable("w", [2], initializer=tf.constant_initializer(0))

loss = tf.reduce_mean(tf.reduce_sum(tf.square(x-w), reduction_indices=1))

grads = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads)

x1 = np.array([[1,1]], dtype=np.float32)

init_all = tf.initialize_all_variables()
session.run(init_all)

prev_w = session.run(w)
grad_w_tensor = session.run(grads[0][0], {x:x1})
grad_w = session.run(grads[0][1], {x:x1})
session.run(train_op, {x:x1})
grad_w = session.run(grads[0][1], {x:x1})
post_w = session.run(w)

trainable_variables = get_name(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
variables = get_name(tf.get_collection(tf.GraphKeys.VARIABLES))

writer = tf.train.SummaryWriter("test_grad_tmp/", session.graph)

print "trainable_variables: {}".format(trainable_variables)
print "variables: {}".format(variables)
print "extra_variables_for_optimizer: {}".format(optimizer.get_slot_names())

print "prev_w: {}".format(prev_w)
print "grad_w_tensor: {}".format(grad_w_tensor)
print "grad_w: {}".format(grad_w)
print "post_w: {}".format(post_w)
