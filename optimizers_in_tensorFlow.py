import tensorflow as tf

# Optimizers in TensorFlow
# GradientDescentOptimizer

loss = 0.2
train_op = tf.compat.v1.train.GradientDescentOptimizer(
    learning_rate=0.001).minimize(loss)

# AdagradOptimizer

train_op = tf.compat.v1.train.AdagradOptimizer(
    learning_rate=0.001, initial_accumulator_value=0.1)

# RMSprop
train_op = tf.compat.v1.train.RMSPropOptimizer(
    learning_rate=0.001, decay=0.9, momentum=0.0, epsilon=1e-10)
# where decay represents α, epsilon represents ϵ, and η represents the learning rate.


# AdadeltaOptimizer
train_op = tf.compat.v1.train.AdadeltaOptimizer(
    learning_rate=0.001, rho=0.95, epsilon=1e-08)


# Adam optimizer
train_op = tf.train.AdamOptimizer(
    learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)

# MomentumOptimizer
train_op = tf.compat.v1.train.MomentumOptimizer(
    learning_rate=0.001, momentum=0.9, use_nesterov=False)
