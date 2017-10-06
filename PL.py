# Artificial Intelligence & Robotics
# Neural Networks

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("", one_hot=True)

print('Test shape:',mnist.test.images.shape)
print('Train shape:',mnist.train.images.shape)
#Neural Network parameters
learningRate=1.5
trainingEpochs=30
batchSize=32
PLbatchSize=256
inputN=784
hiddenN=1000
outputN=10
T1=100
T2=600
a=0
af=3

x=tf.placeholder("float",[None, inputN])
y=tf.placeholder("float",[None, outputN])
PLx=tf.placeholder("float",[None, inputN])
PLy=tf.placeholder("float",[None, outputN])
alpha=tf.placeholder("float",[])
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
pred1 = NN(PLx, weights, biases)


# Cross entropy loss function
Loss = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)),tf.multiply(alpha,tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred1, labels=PLy))))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(Loss)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    t=0
    # Training cycle
    for epoch in range(trainingEpochs):
        avg_cost = 0.
        total_batch = int(10000/batchSize)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batchSize)
            batch_xpred,yy= mnist.validation.next_batch(PLbatchSize)
            batch_ypred = sess.run([pred], feed_dict={x: batch_xpred})
            batch_ypred=batch_ypred[0]
            batch_ypred=batch_ypred.argmax(1)
            kk=np.zeros((PLbatchSize,10))
            for ii in range(PLbatchSize):
                kk[ii,batch_ypred[ii]]=1
            
            if t>T1:
                a=((t-T1)/(T2-T1))*af
            if t>T2:
                a=af
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, Loss], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          PLx:batch_xpred,
                                                          PLy:kk,
                                                          alpha: a})
            t=t+1
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
