import tensorflow as tf


data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x_data = [x_row[0] for x_row in data] # x값
y_data = [y_row[1] for y_row in data] # y값

learning_rate=0.1 # 학습률 지정

a = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
# 0에서 10 사이에서 임의의 수 1개 뽑기
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))
# 0에서 100 사이에서 임의의 수 1개 뽑기

y = a*x_data + b
rmse = tf.sqrt(tf.reduce_mean(tf.square(y-y_data)))

gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)
#rmse 함수의 값을 최소로 하는 경사(기울기)값 찾기

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(gradient_descent)
        if step % 100 ==0:
            print("Epoch:%.f, RMSE=%.04f, 기울기 a = %.4f, y절편 b = %.4f" % (step, sess.run(rmse), sess.run(a), sess.run(b)))
