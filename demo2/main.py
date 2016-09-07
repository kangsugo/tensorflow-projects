#coding:utf-8
import tensorflow as tf


#TensorFlow 程序通常被组织成一个构建阶段和一个执行阶段. 
#在构建阶段, op 的执行步骤 被描述成一个图. 
#在执行阶段, 使用会话执行执行图中的 op.



#--------------------第一步：构建图，构建图中的op节点
#源 op 不需要任何输入, 例如 常量 (Constant) . 源 op 的输出
#被传递给其它 op 做运算.

matrix1 = tf.constant([[3.,2.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1,matrix2)


#--------------------第二步，在会话中启动图
ss = tf.Session()
# 会话负责传递 op 所需的全部输入
# run()的传入参数负责得到输出结果
result = ss.run(product)
print result


#--------------------第三步：关闭会话session,释放资源
ss.close()
