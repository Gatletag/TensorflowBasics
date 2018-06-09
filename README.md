# TensorFlow Basics

Notes on how to use TensorFlow, heavily based on the Stanford University Tensorflow for Deep Learning Research [lecture slides](http://web.stanford.edu/class/cs20si/syllabus.html)

# Table of Contents
1. [Overview of TensorFlow](#lecture1)
2. [TensorFlow Operations](#lecture2)
3. [Basic Models in TensorFlow](#lecture3)
    - [Linear Regression in TensorFlow](#lecture3a)
    - [Logistic Regression in TensorFlow](#lecture3b)
4. *Todo:* Eager execution
5. [Manage Experiments](#lecture5)

## Overview of TensorFlow <a name="lecture1"></a>

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

## TensorFlow Operations <a name="lecture2"></a>

### Visualization with TensorBoards
```python
import tensorflow as tf

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
    print(sess.run(x))
```
Run the command and then open `http://localhost:6006/`
```cmd
> python3 [yourprogram].py
> tensorboard --logdir="./graphs" --port 6006
```

### Constants, Sequences, Variables, Ops

#### Constants
```python
# Example parameter structure for a matrix of constants
tf.constant(
    value,
    dtype=None,
    shape=None,
    name='Const',
    verify_shape=False
)
# Example implementation
a = tf.constant([2,9], name="b")
```
```python
# Example parameter structure for a matrix of zeros or ones
tf.zeros(shape, dtype=tf.float32, name=None)
tf.ones(shape, dtype=tf.float32, name=None)

# Example implementation
a = tf.zeros([2, 3], tf.int32)  # [[0, 0, 0], [0, 0, 0]]
```
```python
# Example parameter structure for a matrix of zeros or ones with same shape as input_tensor
tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)

# Example implementation (input_tensor is [[0, 1], [2, 3], [4, 5]])
tf.zeros_like(input_tensor) # [[0, 0], [0, 0], [0, 0]]
```

```python
# Example parameter structure for a tensor filled with specific value
tf.fill(dims, value, name=None)

# Example implementation
tf.fill([2, 3], 8) # [[8, 8, 8], [8, 8, 8]]
```
#### Sequences
```python
# Constants as sequences
tf.lin_space(start, stop, num, name=None)
tf.lin_space(10.0, 13.0, 4) # [10. 11. 12. 13.]

# Constants as a sequence from a range
tf.range(start, limit=None, delta=1, dtype=None, name='range')
tf.range(3, 18, 3) ==> [3 6 9 12 15]
tf.range(5) ==> [0 1 2 3 4]
```
Some Randomly Generated Constants
```python
tf.random_normal
tf.truncated_normal
tf.random_uniform
tf.random_shuffle
tf.random_crop
tf.multinomial
tf.random_gamma

# Setting the random seed
tf.set_random_seed(seed)
```
#### Variables
```python
# create variables with tf.Variable
s = tf.Variable(2, name="scalar")
m = tf.Variable([[0, 1], [2, 3]], name="matrix")
W = tf.Variable(tf.zeros([784,10]))

# create variables with tf.get_variable (preffered method)
s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())
```

Variables need to be initialised.
- To initialise all variables:
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
```
- To initialise a subset of variables:
```python
with tf.Session() as sess:
    sess.run(tf.variables_initializer([a, b]))
```
- To initialise a single variable:
```python
with tf.Session() as sess:
    sess.run(W.initializer)
```

The variables have just been initialised and still need to be evaluated. To evaluate and get the contents of `W`:
```python
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())
```

Reassigning a variable when requires using the `tf.Variable.assign()` operation
```python
W = tf.Variable(10)
assign_op = W.assign(100) # Remember this is an operation
with tf.Session() as sess:
  sess.run(W.initializer) # W has the value 10 (has not traversed the graph yet though)
  sess.run(assign_op) # W is updated to the value 100 (has not traversed the graph yet though)
  print(W.eval())  # W is now 100
```

### Placeholders
Placeholders are used to assemble the graph first without knowing the values needed for computation.
```python
tf.placeholder(dtype, shape=None, name=None)
# create a placeholder for a vector of 3 elements, type tf.float32
a = tf.placeholder(tf.float32, shape=[3])

b = tf.constant([5, 5, 5], tf.float32)

# use the placeholder as you would a constant or a variable
c = a + b  # short for tf.add(a, b)

with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: [1, 2, 3]}))     # the tensor a is the key, not the string ‘a’
```
The above code uses the `feed_dict` parameter and places the value `[1,2,3]` into `a`. The resulting output is `[6,7,8]`

Other Variables can be fed into the graph in the same way, they don't need to be Placeholder values. Check if a tensor is feedable with `tf.Graph.is_feedable(tensor)`
```python
# create operations, tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.multiply(a, 3)

with tf.Session() as sess:
    # compute the value of b given a is 15
    sess.run(b, feed_dict={a: 15})                 # >> 45
```

### Avoid Lazy Loading
```
1. Separate definition of ops from computing/running ops
2. Use Python property to ensure function is also loaded once the first time it is called
```

## Basic Models in TensorFlow<a name="lecture3"></a>

### Linear Regression in TensorFlow <a name="lecture3a"></a>
For a linear regression, we want to model the linear relationship between:
- dependent variable Y
- explanatory variables X

```
World Development Indicators dataset
X: birth rate
Y: life expectancy
190 countries
GOAL: Find a linear relationship between X and Y to predict Y from X
```
**Model**
```
Inference: Y_predicted = w * X + b
Mean squared error: E[(y - y_predicted)^2]
```

[**Code example**](!examples/03_linreg_starter.py)

- `examples/03_linreg_starter.py`

```
> python3 03_linreg_starter.py
> tensorboard --logdir='./graphs'
```

**Some Code notes from Example**
```python
tf.data.Dataset.from_tensor_slices((features, labels))

dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
print(dataset.output_types)        # >> (tf.float32, tf.float32)
print(dataset.output_shapes)        # >> (TensorShape([]), TensorShape([]))

tf.data.TextLineDataset(filenames)
tf.data.FixedLengthRecordDataset(filenames)
tf.data.TFRecordDataset(filenames)
```
**Making an Iterator**
```python
iterator = dataset.make_one_shot_iterator()
# Iterates through the dataset exactly once. No need to initialization.

iterator = dataset.make_initializable_iterator()
#Iterates through the dataset as many times as we want. Need to initialize with each epoch.
```

**Performing Operations on the dataset**
```python
dataset = dataset.shuffle(1000)
dataset = dataset.repeat(100)
dataset = dataset.batch(128)
dataset = dataset.map(lambda x: tf.one_hot(x, 10))  # convert each elem of dataset to one_hot vector
```

**Specify if you want to train the variable**
```python
tf.Variable(initial_value=None, trainable=True,...)
```

**List of Optimizers in TensorFlow**
```python
tf.train.GradientDescentOptimizer
tf.train.AdagradOptimizer
tf.train.MomentumOptimizer
tf.train.AdamOptimizer
tf.train.FtrlOptimizer
tf.train.RMSPropOptimizer
```

### Logistic Regression in TensorFlow <a name="lecture3b"></a>

```
MNIST Database
X: image of a handwritten digit
Y: the digit value
GOAL: Recognize the digit in the image
```
**Model**
```
Inference: Y_predicted = softmax(X * w + b)
Cross entropy loss: -log(Y_predicted)
```

[**Code example**](!examples/03_logreg_starter.py)

- `examples/03_logreg_starter.py`

```
> python3 03_logreg_starter.py
> tensorboard --logdir='./graphs'
```

**Mutliple Iterators**
```python
iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data

# During Training
sess.run(train_init)               # use train_init during training loop
```

## Manage Experiments <a name="lecture5"></a>

### Word Embeddings in TensorFlow <a name="lecture5a"></a>

**Word Embedding**

*Representing a word by means of its neighbors*
- Distributed representation
- Continuous values
- Low dimension
- Capture the semantic relationships between words

#### CBOW vs Skip-Gram

**CBOW**

- predicts center words from context words
- many-to-one
- better for smaller datasets (entire context as one observation)

**Skip-gram**
- predicts source context-words from the center words
- one-to-many
- treats each context-target pair as new observation

**Embedding Lookup**
```python
tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)
```
**NCE Loss (Alternative to Softmax)**
```python
tf.nn.nce_loss(
    weights,
    biases,
    labels,
    inputs,
    num_sampled,
    num_classes,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits=False,
    partition_strategy='mod',
    name='nce_loss'
)
```

**Phase 1: Assembling the Graph**
1. Import data (with tf.data or placeholders)
2. Define the weights
3. Define the inference model
4. Define loss function
5. Define optimizer

**Model Reuse with Object Oriented Programming**

Example from `examples/04_word2vec_visualize.py` shows a good end to end approach
```python
class SkipGramModel:
    """ Build the graph for word2vec model """
    def __init__(self, params):
        pass

    def _import_data(self):
        """ Step 1: import data """
        pass

    def _create_embedding(self):
        """ Step 2: define weights. In word2vec, it's actually the weights that we care about """
        pass

    def _create_loss(self):
        """ Step 3 + 4: define the inference + the loss function """
        pass

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        pass
```

### Name scope
Group nodes together with tf.name_scope(name)

```python
with tf.name_scope('data'):
    iterator = dataset.make_initializable_iterator()
    center_words, target_words = iterator.get_next()
with tf.name_scope('embed'):
    embed_matrix = tf.get_variable('embed_matrix',
                                    shape=[VOCAB_SIZE, EMBED_SIZE], ...)
    embed = tf.nn.embedding_lookup(embed_matrix, center_words)
with tf.name_scope('loss'):
    nce_weight = tf.get_variable('nce_weight', shape=[VOCAB_SIZE, EMBED_SIZE], ...)
    nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, …)
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
```

### Variable scope

Use `get_variable` as then you will know if you are reusing existing graph or creating a new tensor for each input. Easier to debug.
```python
tf.get_variable(<name>, <shape>, <initializer>)

# If a variable with <name> already exists, reuse it
# If not, initialize it with <shape> using <initializer>

def two_hidden_layers(x):
    assert x.shape.as_list() == [200, 100]
    w1 = tf.get_variable("h1_weights", [100, 50], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("h1_biases", [50], initializer=tf.constant_initializer(0.0))
    h1 = tf.matmul(x, w1) + b1
    assert h1.shape.as_list() == [200, 50]  
    w2 = tf.get_variable("h2_weights", [50, 10], initializer=tf.random_normal_initializer())
    b2 = tf.get_variable("h2_biases", [10], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(h1, w2) + b2
    return logits

with tf.variable_scope('two_layers') as scope:
    logits1 = two_hidden_layers(x1)
    scope.reuse_variables() # note this line to ensure same tensors are used.
    logits2 = two_hidden_layers(x2)

```

Alternatively creating scope (same as above code):
```python
def fully_connected(x, output_dim, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable("weights", [x.shape[1], output_dim], initializer=tf.random_normal_initializer())
        b = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b

def two_hidden_layers(x):
    h1 = fully_connected(x, 50, 'h1')
    h2 = fully_connected(h1, 10, 'h2')

with tf.variable_scope('two_layers') as scope:
    logits1 = two_hidden_layers(x1)
    logits2 = two_hidden_layers(x2)
```

### Saving Sessions
```python
tf.train.Saver.save(sess, save_path, global_step=None...)
tf.train.Saver.restore(sess, save_path)
```

**Example**

```python
# Within the Model Class
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

# define model
model = SkipGramModel(params)

# create a saver object
saver = tf.train.Saver()

with tf.Session() as sess:
    for step in range(training_steps):
        sess.run([optimizer])

        # save model every 1000 steps
        if (step + 1) % 1000 == 0:
            saver.save(sess, 'checkpoint_directory/model_name', global_step=step)
```

**Saving specific variables**
```python
#Can also choose to save certain variables
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')

#You can save your variables in one of three ways:
saver = tf.train.Saver({'v1': v1, 'v2': v2})
saver = tf.train.Saver([v1, v2])
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]}) # similar to a dict
```
**Restoring session variables**
```python
saver.restore(sess, 'checkpoints/name_of_the_checkpoint')

# RESTORING LATEST CHECKPOINT
# check if there is checkpoint
ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

# check if there is a valid checkpoint path
if ckpt and ckpt.model_checkpoint_path:
     saver.restore(sess, ckpt.model_checkpoint_path)
```

### Creating summaries
```python
# Step 1: Creating Summaries
with tf.name_scope("summaries"):
    tf.summary.scalar("loss", self.loss)
    tf.summary.scalar("accuracy", self.accuracy)            
    tf.summary.histogram("histogram loss", self.loss)
    summary_op = tf.summary.merge_all()

# Step 2: Running Summaries
loss_batch, _, summary = sess.run([loss, optimizer, summary_op])

# Step 3: Writing summaries to file
writer.add_summary(summary, global_step=step)
```

### Controlling Randomness
```python
my_var = tf.Variable(tf.truncated_normal((-1.0,1.0), stddev=0.1, seed=0))
tf.set_random_seed(2)
```

### Gradients
```python
x = tf.Variable(2.0)
y = 2.0 * (x ** 3)
z = 3.0 + y ** 2
grad_z = tf.gradients(z, [x, y])
with tf.Session() as sess:
    sess.run(x.initializer)
    print(sess.run(grad_z)) # >> [768.0, 32.0]
# 768 is the gradient of z with respect to x, 32 with respect to y

# Gradient Computation
tf.gradients(ys, xs, grad_ys=None, ...)
tf.stop_gradient(input, name=None)
# prevents the contribution of its inputs to be taken into account
tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)
tf.clip_by_norm(t, clip_norm, axes=None, name=None)
```
