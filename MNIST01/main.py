#coding:utf-8
import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#****************************************1.图的构建节点阶段
#图片 * 权重 = evidences
#矩阵相乘： x,784 * 784,10 = x,10 得到x张图片10个数字中各个的evidences
#对于一张图片 : 1 * 724 . 724 * 10 = 1 * 10,结果是10维向量，对应是每一个数的概率

x=tf.placeholder("float",[None,784])#x是一张图片对应的724维像素【1,2,3，...724】

w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,w)+b)#y是这张图片对应的便签
y_ = tf.placeholder("float",[None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# TensorFlow拥有一张描
# 述你各个计算单元的图,它可以自动地使用反向传播算法(backpropagation algorithm)来有效地确定你的变量是
# 如何影响你想要最小化的那个成本值的。然后,TensorFlow会用你选择的优化算法来不断地修改变量以降低成本
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


#****************************************2.图的操作节点阶段
init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    # 该循环的每个步骤中,我们都会随机抓取训练数据中的100个批处理数据点,然后我们用这些数据点作为参数替换
    # 之前的占位符来运行 train_step 。
    # 也就是前面定义的两个placeholder : x, y_, 这种映射是自动对应的



#评估我们的模型
# 首先让我们找出那些预测正确的标签。 tf.argmax
# 是一个非常有用的函数, 它能给出某个tensor对象在某一维上
# 的其数据最大值所在的索引值。由于标签向量是由0, 1
# 组成, 因此最大值1所在的索引位置就是类别标签, 比如
# tf.argmax(y, 1)
# 返回的是模型对于任一输入x预测到的标签值, 而tf.argmax(y_, 1)
# 代表正确的标签, 我们可以
# 用tf.equal来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

