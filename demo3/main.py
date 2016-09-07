#coding:utf-8
import tensorflow as tf
#实现加法器功能


#************************************1.图的构建节点阶段
#****************************************************
# 创建一个变量op, 初始化为标量 0
state = tf.Variable(0,name="counter")

# 创建一个源数据op, 其作用是使 state 增加 1
one = tf.constant(1)

#以下是行为操作op（operation）
new_value = tf.add(state,one)
update = tf.assign(state,new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()





#************************************2.图的操作节点阶段
#****************************************************
# 启动图, 运行 op
with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(state)
    for _ in range(3):
        sess.run(update)
        print sess.run(state)

