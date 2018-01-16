import tensorflow as tf
import pickle as pk
from sklearn.model_selection import train_test_split
import numpy as np
from mlxtend.preprocessing import one_hot

mnist = pk.load(open("hot_dogs\\allData.p",'rb'))
labels = mnist[1]
labels = one_hot(labels)
data = mnist[0]

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])
	
	
def weight_variable(shape): # function to create weight
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape): # function for bias variable.
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d (x,W): # compute convolution / convoultion layer part ~ where the mask and the weight are multiplied etc to get activation for that masks area to be fed into feature map
    return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding= 'SAME')

def max_pool_2x2(x): # pooling layer to pool feature map via 2*2 mask, basically halfing the feature map by taking max over 2*2 masks over whole image
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1], padding= 'SAME')
    
	
# first convolutional layer
# consist of convolution followed bt max pooling.
# convolution will compute 32 features for each 5x5 patch. its weight tensor shape = [5,5,1,32]
# first two dimnesion are filter size, next is number of input channels = 1, last = number of output channels/features = 32
# also a bias chanel

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# to apply the first layer we first reshape x to a 4d tensor, 2nd and 3rd dims are image width and height, final dim = number of colur channels
x_image = tf.reshape(x,[-1,28,28,1])


# then convolve the image with the weight tensors adding bias , apply the relu function and finally the max pool = 2x2 method reduces the image to 14x14.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convultional layer
# will have 64 features for every 5x5 patch.

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # this concultion takes output of first layer which is pooled feature map = h_pool1
h_pool2 = max_pool_2x2(h_conv2) # output of 2nd layer is the pooled feature map h_pool2, images now down to 7x7 size

# imags now down to 7x7 size, 
# computationally possible to add fully connected dense layer with 1024 neuron to allow processing the entire image. 
# reshape the tensor from the pooling layer into batch of vectors, multiply by weight matrix, add bias and apply RELU.

W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat =tf.reshape(h_pool2,[-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

# dropout - to reduce overfitting.
#create placeholder for the prob that a neurons output is kept during dropout.
# allows s to turn on dropout during training and off during testing.

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#readout layer, finally add a layer. softmax layer essentially which just does (x*W) + b 

W_fc2 = weight_variable([1024,2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: x_train, y_:y_train, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: x_train, y_:y_train, keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={x: x_train, y_: y_train, keep_prob: 1.0}))