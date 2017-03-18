#-*- coding: utf-8 -*-　
#引用必要函數
import tensorflow as tf
import numpy as np

# creat data
#創造 亂數100個亂數 值介於0~1之間
x_data = np.random.rand(100).astype(np.float32)

#訂出要學習的函數  這裡要學的是 weight:0.1  biases:0.3
y_data = np.sin(x_data)  #x_data*0.1+0.3

###creat tensorflow structure start###
#定立 weights 的 範圍 和初始化
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

#定立 給 tensorflow 學習的函數
y = Weights*x_data + biases

#建立 loss 規則
loss = tf.reduce_mean(tf.square(y-y_data))

#選擇學習機制
optimizer = tf.train.GradientDescentOptimizer(0.5)  #learning rate

#像tensorflow 說 訓練規則就是把 loss 減到最小 最好是0
train = optimizer.minimize(loss)

#初始化所有變數
init = tf.initialize_all_variables()

###creat tensorflow structure end###
#上面都是建立規則
#*****************************************************開始訓練
#建立sess
sess = tf.Session()

#記得初始化
sess.run(init)     #Very important

"""
跑 for 迴圈 更新wight201次
然後每訓練20次 印出一次weight
"""
for step in range(201):
    sess.run(train)
    if step % 20 ==0:
        print(step,sess.run(Weights),sess.run(biases))