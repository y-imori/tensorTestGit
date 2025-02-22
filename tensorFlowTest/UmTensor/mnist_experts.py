from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#初期化
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#初期化
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#入力値xと重みWを引数としてとる
def conv2d(x,W):
    #2次元の込畳みこみ
    #1pxずつずらしてずらして適用する
    #入力と同じサイズの出力
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#畳み込まれたデータの中で
#2px*2pxの中で一番サイズの大きなデータを出力として返す
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#一番目の重みとバイアス
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
#Xは複数あるので、Xの数文の入れ物
#-1は実行時に自動で割り当てるフラグ
x_image = tf.reshape(x, [-1, 28, 28, 1])
#畳み込んだ結果に対して、正規化関数のreluを適用しプーリング
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#全結合
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#dropout過学習に備える
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#予測値と実測値の誤差を最小にしていく
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#正解率：確率の高いものと実測値が正しいか
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print("test sccuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))