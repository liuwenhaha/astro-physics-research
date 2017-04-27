3/6/2017
1. Install TensorFlow
https://www.tensorflow.org/install/install_linux
-> Install Nvidia GPU base
    http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A
    CUDA Toolkit
    need to download https://developer.nvidia.com/cuda-downloads
        backup in hardware
    CUDNN
        https://askubuntu.com/questions/767269/how-can-i-install-cudnn-on-ubuntu-16-04

3/13/2017
GET STARTED
1. Getting Started With Tensorflow
https://www.tensorflow.org/get_started/get_started
Computational Graph
    computational graph: series of TF operations arranged into a graph of nodes
    tf.Session(): computational graph run within it, and nodes are evaluated
        sess.run()
    TensorBoard can display picture of computational graph
    tf.placeholder(tf.float32): parameterized to accept external inputs; is a promise to provide a value later
    tf.Variable: trainable parameters in a graph
    tf.constant: constants never change value
    To initialize all variables in TF program:
        init = tf.global_variables_initializer()
        sess.run(init)
        init is a handle to TF sub-graph that initializes all global variables
    A model with placeholder to provide desired values, also with loss function
    tf.square()
    tf.reduce_sum()
    tf.assign()
tf.train() API
    optimizers: slowly change each variable to minimize loss function
    gradient descent: the simplest optimizer
    TensorFlow automatically produce derivatives given only a description o the model using function tf.gradients
    tf.train.GradientDescentOptimizer(learning_rate)
    optimizer.minimize(loss)

3/14/2017
2. MNIST for ML Beginners
https://www.tensorflow.org/get_started/mnist/beginners
Softmax Regressions
    assign probabilities to an object being one of several different things
    two steps:
         first we add up the evidence of our input being in certain classes
         we do a weighted sum of the pixel intensities. The weight is negative if that pixel having a high intensity is evidence against the image being in that class, and positive if it is evidence in favor
         extra evidence called a bias, some things are more likely independent of the input.
         then we convert that evidence into probabilities
     softmax is serving as an "activation" or "link" function, shaping the output of our linear function into the form we want
 Training
    cross-entropy




3. Deep MNIST for Experts
    TensorFlow relies on a highly efficient C++ backend to do its computation. The connection to this backend is called a session. The common usage for TensorFlow programs is to first create a graph and then launch it in a session.
    To do efficient numerical computing in Python, we typically use libraries like NumPy that do expensive operations such as matrix multiplication outside Python, using highly efficient code implemented in another language. Unfortunately, there can still be a lot of overhead from switching back to Python every operation. This overhead is especially bad if you want to run computations on GPUs or in a distributed manner, where there can be a high cost to transferring data.
    TensorFlow also does its heavy lifting outside Python, but it takes things a step further to avoid this overhead. Instead of running a single expensive operation independently from Python, TensorFlow lets us describe a graph of interacting operations that run entirely outside Python. This approach is similar to that used in Theano or Torch.
    The role of the Python code is therefore to build this external computation graph, and to dictate which parts of the computation graph should be run.
Multilayer Convolutional Network
    Initialize with noise for symmetry breaking and prevent 0 gradient
    ReLU -> small positive for avoiding "dead neurons"
    For this small convolutional network, performance is actually nearly identical with and without dropout. Dropout is often very effective at reducing overfitting, but it is most useful when training very large neural networks.









4. TensorFlow Mechanics 101
Goal is to use TensorFlow to train and evaluate a simple feed-forward neural network for MNIST.
Tutorial Files
Prepare the Data
Build the Graph
    Inference
    Loss
    Training
Train the Model
    The Graph
    The Session
    Train Loop
Evaluate the Model
    Build the Eval Graph
    Eval Output


5. tf.contrib.learn Quickstart

6. Building Input Functions with tf.contrib.learn

7. TensorBoard: Visualizing Learning

8. TensorBoard: Embedding Visualization

9. TensorBoard: Graph Visualization

10. Logging and Monitoring Basics with tf.contrib.learn



PROGRAMMER'S GUIDE
1. Threading and Queues

2. Sharing Variables

3. TensorFlow Version Semantics

4. TensorFlow Dta Versioning: GraphDefs and Checkpoints

5. Supervisor: Training Helper for Days-Long Trainings

6. TensorFlow Debugger(tfdbg)

7. Command-Line-Interface Tutorial: MNIST

8. How to Use TensorFlow Debugger(tfdbg) with tf.contrib.learn

9. Exporting and Importing a MetaGraph

10. Tensor Ranks, Shapes, and Types

11. Variables: Creation, Initialization, Saving, and Loading




TUTORIALS
1. Mandelbrot Set

2. Partial Differential Equations

3. Convolutional Neural Networks

4. Image Recognition

5. How to Retrain Inception's Final Layer for New Categories

6. Vector Representation of Words

7. Recurrent Neural Networks

8. Sequence-to-Sequence Models

9. A Guide to TF Layers: Building a Convolutional Neural Network

10. Large-scale Linear Models with TensorFlow

11. TensorFlow Linear Model Tutorial

12. TensorFlow Wide & Deep Learning Tutorial

13. Using GPUs




