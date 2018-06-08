# Tensorflow Basics

Notes on how to use Tensorflow, heavily based on the Stanford University Tensorflow for Deep Learning Research [lecture slides](http://web.stanford.edu/class/cs20si/syllabus.html)

# Table of Contents
1. [Overview of Tensorflow](#lecture1)


## Overview of Tensorflow <a name="lecture1"></a>

### Graphs and Sessions

#### Tensor

A tensor is an n-dimensional array

```
0-d tensor = scalar
1-d tensor = vector
2-d tensor = matrix
...
```

#### Data Flow Graph

**Phase 1: assemble a graph**

```python
import tensorflow as tf
a = tf.add(3,5)
print(a)
```
Output:
```
>>> Tensor("Add:0", shape=(), dtype=int32)
```

**Phase 2: use a session to execute operations in the graph**

```python
sess = tf.Session()
print(sess.run(a))
sess.close()
```
Output:
```
>>> 8
```
The session evaluates the graph to fetch the value of `a`. 
*Session* 
- encapsulates the environment in which Operation objects are executed
- allocates memory to store the value of variables
*Tensor* 
- the data that is evaluated

Alternatively, the following code starts the session and closes the session after completion
```python
with tf.Session() as sess:
  print(sess.run(a))
```

#### Subgraphs
```python
x = 2
y = 3
add_op = tf.add(x,y)
mul_op = tf.mutliply(x,y)
useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op, mul_op)

with tf.Session() as sess:
  z, not_useless = sess.run([pow_op, useless])
```
In the above example we want the session to give back (fetch) a list of tensors. We get the `pow_op` and `useless` which are saved to `z` and `not_useless` respectively.

The parameters for running the session graph are:
```python
sess.run(fetches, feed_dict=None, options=None, run_metadata=None)
```
Further details about parameters will be discussed later. So far we have just used the `fetches` and specified it to `[pow_op, useless]` to evaluate the graph and save to `z` and `not_useless`.

#### Distributed Computation

The graphs can be speciified where to perform the computation using `with tf.device('/gpu:0'):`
```python
with tf.device('/gpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
  c = tf.multiply(a, b)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(sess.run(c))
sess.close()
```
