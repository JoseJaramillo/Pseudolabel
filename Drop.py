# Artificial Intelligence & Robotics
# Neural Networks

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("", one_hot=True)

print('Test shape:',mnist.test.images.shape)
print('Train shape:',mnist.train.images.shape)
#Neural Network parameters
learningRate=1.5
trainingEpochs=30
batchSize=32
inputN=784
hiddenN=500
outputN=10
print('NN+Dropout ',hiddenN)
x=tf.placeholder("float",[None, inputN])
y=tf.placeholder("float",[None, outputN])
training=tf.placeholder("float",[])

def NN(x,weight,bias,training):
    
#        print( 'x:', x.get_shape(), 'W1:', weight['h1'].get_shape(), 'b1:', bias['b1'].get_shape())        
    # Hidden layer 
    HL = tf.add(tf.matmul(x, weights['h1']), biases['b1']) # HL=(x*h1)+b1
    HL = tf.nn.relu(HL)  
    if training==1:
        HL = tf.nn.dropout(HL, keep_prob=0.5)                             # sigmoid(HL)
    # Output layer with linear activation
#        print( 'HL:', HL.get_shape(), 'W2:', weight['out'].get_shape(), 'b2:', bias['out'].get_shape())        
    out_layer = tf.matmul(HL, weights['out']) + biases['out'] # Out=(HL*h1)+b1
#        print('out_layer:',out_layer.get_shape())
    
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

pred = NN(x, weights, biases,training)

# Cross entropy loss function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    t=1
    # Training cycle
    for epoch in range(trainingEpochs):
        avg_cost = 0.
        total_batch = int(100/batchSize)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batchSize)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          training: t})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % 1 == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
#       print("Optimization Finished!")
    t=0
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    print (accuracy.eval({x: mnist.test.images, y: mnist.test.labels, training:t}))
