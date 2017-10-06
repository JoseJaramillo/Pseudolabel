# Artificial Intelligence & Robotics
# Neural Networks

import tensorflow as tf
import gzip
import pickle

#Load the Mnist dataset
f = gzip.open('mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("", one_hot=True)

print('Test shape:',mnist.test.images.shape)
print('Train shape:',mnist.train.images.shape)
#Neural Network parameters
learningRate=1.5
trainingEpochs=30
batchSize=100
inputN=784
hiddenN=1000
outputN=10

x=tf.placeholder("float",[None, inputN])
y=tf.placeholder("float",[None, outputN])

def NN(x,weight,bias):
    
    print( 'x:', x.get_shape(), 'W1:', weight['h1'].get_shape(), 'b1:', bias['b1'].get_shape())        
    # Hidden layer 
    HL = tf.add(tf.matmul(x, weights['h1']), biases['b1']) # HL=(x*h1)+b1
    HL = tf.nn.sigmoid(HL)                               # sigmoid(HL)
    
    # Output layer with linear activation
    print( 'HL:', HL.get_shape(), 'W2:', weight['out'].get_shape(), 'b2:', bias['out'].get_shape())        
    out_layer = tf.matmul(HL, weights['out']) + biases['out'] # Out=(HL*h1)+b1
    print('out_layer:',out_layer.get_shape())
    
    return out_layer

#initialize weights and biases

weights = {
    'h1': tf.Variable(tf.random_normal([inputN, hiddenN])),    #784x256
    'out': tf.Variable(tf.random_normal([hiddenN, outputN]))  #256x10
}
biases = {
    'b1': tf.Variable(tf.random_normal([hiddenN])),             #256x1
    'out': tf.Variable(tf.random_normal([outputN]))              #10x1
}

pred = NN(x, weights, biases)

# Cross entropy loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(trainingEpochs):
        avg_cost = 0.
        total_batch = int(10000/batchSize)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batchSize)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % 1 == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
