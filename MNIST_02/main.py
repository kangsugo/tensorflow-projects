#coding:utf-8
#使用多层卷积神经网络提高MNIST准确率
import tensorflow as tf
import numpy
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class MNIST_with_CNN:
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        # 截断正态分布
        # shape ： 输出张量的维度（形状）
        # stddev ：分布的标准差。一个较大的标准差，代表大部分数值和其平均值之间差异较大；一个较小的标准差，代表这些数值较接近平均值。
        return tf.Variable(initial)

    def bias_varible(self,shape):
        initial = tf.constant(0.1,shape=shape)
        # 函数说明
        # 第一个参数是用于初始化的参数，第二个参数是将要把初始化参数放入的张量的维度格式

        return tf.Variable(initial)


    def conv2d(self,x,w):
        # 我们的卷积使用1步长(stridesize), 0边距(paddingsize)的模板,
        #（步长为1，填充为0，可以保证输入输出大小相同，相当于每一个元素卷积后的结果结果都能占一个位置）
        # 保证输出和输入是同一个大小。
        return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


if __name__=="__main__":
    tool = MNIST_with_CNN()

    x = tf.placeholder("float", [None, 784])  # x是一张图片对应的724维像素【1,2,3，...724】

    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, w) + b)  # y是这张图片对应的便签
    y_ = tf.placeholder("float", [None, 10])

    #------------------------------>第一层卷积
    # 卷积在每个5x5的patch中算出32（两个矩阵：32 = 4*4 *2）个特征。
    #(应为卷积得到的是数值：5 * 5的矩阵和5 * 5的矩阵卷积结果为一个值。即对应位置相乘后求和。)
    # 卷积的权重张量形状是[5, 5, 1, 32], 前两个维度是patch的大小, 接着是输入的通道数目, 最后是输出的
    # 通道数目。 而对于每一个输出通道都有一个对应的偏置量。
    w_conv1 = tool.weight_variable([5,5,1,32])#<------->
    b_conv1 = tool.bias_varible([32])

    #为了用这一层,我们把 x 变成一个4d向量,其第2、第3维对应图片的宽、高,最后一维代表图片的颜色通道数(因
    #为是灰度图所以这里的通道数为1,如果是rgb彩色图,则为3)。
    x_image = tf.reshape(x,[-1,28,28,1])#将28 * 28的图片转换成4d向量 #<------->
    h_conv1 = tf.nn.relu(tool.conv2d(x_image,w_conv1) + b_conv1)
    h_pool1 = tool.max_pool_2x2(h_conv1)


    #------------------------------>第二层卷积
    w_conv2 = tool.weight_variable([5,5,32,64])
    b_conv2 = tool.bias_varible([64])

    h_conv2 = tf.nn.relu(tool.conv2d(h_pool1,w_conv2) + b_conv2)
    h_pool2 = tool.max_pool_2x2(h_conv2)

    #------------------------------>密集连接层
    #将图片转换成像素向量
    # 此时图片转化为7*7
    # 注意：在本程序中，卷积由于步长,padding等的设置，不改变图片大小。而池化会改变大小

    # 图片尺寸减小到7x7, 我们加入一个有1024个神经元的全连接层
    # 注意：此处为何行的个数是1024，看起来颠倒了？
    # 因为权值是右乘，左边是1 * 7 * 7 * 64，右边是7 * 7 * 64 * 1024，结果是1 * 1024，一张图片分到1024个神经元
    w_fc1 = tool.weight_variable([7*7*64,1024])
    b_fc1 = tool.bias_varible([1024])

    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])#此时将图片转化成行向量作为张量的第二维，-1所在的维度就是图片的个数
    # 函数说明
    # reshape（t, [-1]）:将张量展开成一行。
    # reshape（t, [a, -1]）:如果有一维度是 - 1，则说明这一维度的取值是元素总数 / a, 即保证张量的size保持不变。

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    #------------------------------>输出层
    w_fc2 = tool.weight_variable([1024,10])
    b_fc2 = tool.bias_varible([10])
    
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print "step %d, training accuracy %g" % (i, train_accuracy)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})







