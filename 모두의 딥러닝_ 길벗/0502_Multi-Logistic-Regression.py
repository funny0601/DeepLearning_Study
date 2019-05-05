import tensorflow as tf
import numpy as np

x_data = np.array([[2, 3], [4, 3], [6, 4], [8, 6], [10, 7], [12, 8], [14, 9]])
# 속성 데이터
y_data = np.array([0, 0, 0, 1, 1, 1, 1]).reshape(7, 1)
# class 데이터

new_x = np.array([7, 6.]).reshape(1, 2)

# reshape 후 [[0], [0], [0], [1], [1], [1], [1]] 로 변경

X = tf.placeholder(tf.float64, shape = [None, 2])
Y = tf.placeholder(tf.float64, shape = [None, 1])

# a, b 임의로 정함
a = tf.Variable(tf.random_uniform([2, 1], dtype = tf.float64))
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))

# sigmoid 함수
y = tf.sigmoid(tf.matmul(X, a)+b)

# 오차 구하는 함수
loss = -tf.reduce_mean(Y*tf.log(y)+(1-Y)*tf.log(1-y))

learning_rate =0.1

gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# predicted = tf.cast(y>0.5, dtype = tf.float64) # 0.5 초과일 경우 1, 아니면 0
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype= tf.float64))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(300001):
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_descent], feed_dict={X:x_data, Y:y_data})
        if(i+1) % 300 ==0:
            print("step=%d, a1=%.4f, a2=%.4f, b=%.4f, loss=%.4f" % (i+1, a_[0], a_[1], b_, loss_))

    new_y = sess.run(y, feed_dict={X: new_x})

# epoch: 300000, loss: 0.0007
print("합격 가능성: %6.2f %%" %(new_y*100))