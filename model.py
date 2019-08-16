import tensorflow as tf
import chinese_word2vec as cwv
import numpy as np
import data_process as dp
from sklearn import metrics

# 设置超参数
num_classes = 4
batch_size = 128
embedding_size = 100
lr = 0.1
filter_size = [2, 3, 4]
decay_steps = 32
decay_rate = 0.8
decay_rate_big = 0.5
clip_gradients = 5
optimizer_type = 1
max_model_num = 5
epoch = 100
# dropout_keep_prob = 1.0


# 获得数据
content_word2id, one_hot_label, word2id = dp.run()
print('content_word2id', content_word2id)
print('word2id', word2id)
vocab_size = len(word2id)

train_remainder = len(content_word2id) % batch_size
# 保证是batch的整数倍
train_data = content_word2id + content_word2id[0:batch_size-train_remainder]
train_label = one_hot_label + one_hot_label[0:batch_size-train_remainder]
print('train_data', train_data)



# 对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
def pad_sentence_batch(sentence_batch, pad_int):
    """Pad the mini batch, so that each input/output has the same length
    Args:
        sentence_batch: mini batch to get padded.
        pad_int: an integer representing the symbol of <PAD>
    Returns:
        Padded mini-batch
    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# 定义生成器，用来获取batch
def get_batch(sources, label, pad_int, batch_size):
    """Generator to generating the mini batches for training and testing
    """
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        label_batch = label[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, pad_int))
        # 记录每条记录的长度
        source_lengths = []
        for source in pad_sources_batch:
            source_lengths.append(len(source))
        yield np.asarray(pad_sources_batch, np.float32), np.asarray(label_batch, np.int32), np.asarray(source_lengths, np.int32)


# 定义计算图
graph = tf.Graph()

with graph.as_default():
    # define the global step of the graph
    global_step = tf.train.create_global_step(graph)
    # 定义占位符
    input_x = tf.placeholder(tf.int32, [batch_size, None], name='input_x')
    input_y = tf.placeholder(tf.int32, [batch_size, num_classes], name='input_y')
    # L = tf.placeholder(dtype=tf.float32, name="learning_rate")
    L = tf.Variable(lr, trainable=False, name='learning_rate')
    sequence_length = tf.placeholder(tf.int32, (batch_size,), name="sequence_length")
    max_target_sequence_length = tf.reduce_max(sequence_length, name='max_target_len')

    with tf.variable_scope("embedding", dtype=tf.float32):
        embedding = tf.get_variable('embedding', shape=[vocab_size, embedding_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.01))  # vocab_size 个 大小 为 embedding_size 的词向量随机初始化
        embedded_wrods = tf.nn.embedding_lookup(embedding, input_x)
        print('embedding', embedding)
        print('embedded_wrods', embedded_wrods)
        print('input_x', input_x)
    print('embedded_wrods.shape', embedded_wrods.shape)
    # 对输入数据input_扩维
    x_expand = tf.expand_dims(embedded_wrods, -1)
    print('x_expand.shape', x_expand.shape)
    # CNN
    pooling_output = []
    for i, filter_size_j in enumerate(filter_size):
        with tf.name_scope('convolution_pooling_%s' % filter_size_j):
            # filter_ = tf.get_variable('filter_%s' % filter_size_j,
            #                           [filter_size_j, embedding_size, 1, num_filters],
            #                           initializer=tf.truncated_normal_initializer(stddev=0.01))  # 【高，宽，通道数，卷积核数】
            # conv = tf.nn.conv2d(x_expand, filter_, strides=[1, 1, 1, 1], padding='VALID',
            #                     name='conv')  # 'SAME'为等长卷积， 'VALID'为窄卷积， strides卷积步长
            #                print('conv: ', conv)

            # model_b = tf.get_variable('model_b_%s' % filter_size_j, [num_filters])
            # model_h = tf.nn.relu(tf.nn.bias_add(conv, model_b), 'relu')

            # pooling = tf.nn.max_pool(conv1, ksize=[1, sequence_length - filter_size_j + 1, 1, 1],
            #                          strides=[1, 1, 1, 1], padding='VALID', name='pool')
            # [batchsize, height, width,filters]
            conv1 = tf.layers.conv2d(x_expand, filters=1, kernel_size=[2, embedding_size], padding='same', activation=tf.nn.relu)
            # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            print(i, conv1.shape)
            pooling_output.append(conv1)  # [3,batchsize, height, width,filters]
    print('pooling_output', pooling_output)
    pool_concat = tf.concat(pooling_output, 3)
    # print(333,pool_concat.shape)
    conv_shape = tf.shape(pool_concat)
    # [-1,len(filter_size)*sequence_length*embedding_size]
    # pool_concat_flat = tf.reshape(pool_concat, [-1, num_filters * len(filter_size)])
    # conv_shape[1]*conv_shape[2]*conv_shape[3]
    # pool_concat_flat = tf.reshape(pool_concat, [batch_size, -1])
    l_ = max_target_sequence_length * embedding_size*3
    pool_concat_flat = tf.reshape(pool_concat, shape=(batch_size, -1, embedding_size*3))
    out = tf.reduce_mean(pool_concat_flat, axis=1)
    # pool_concat_flat = tf.reshape(pool_concat, shape=(batch_size, l_))
    # with tf.name_scope('dropout'):
    #     m_dropout = tf.nn.dropout(out, keep_prob=dropout_keep_prob)

    # dropout
    # h_fc_drop1 = tf.nn.dropout(dense1, keep_prob)
    # print(777,pool_concat_flat.shape)
    logits = tf.layers.dense(out, num_classes)
    # logits = tf.layers.dense(inputs=pool_concat_flat, units=num_classes, activation=None,
    #                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
    #                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y))
    learning_rate = tf.train.exponential_decay(L, global_step, decay_steps,
                                               decay_rate, staircase=True)
    optimizer_collection = {0: tf.train.GradientDescentOptimizer(learning_rate),
                            1: tf.train.AdamOptimizer(learning_rate),
                            2: tf.train.RMSPropOptimizer(learning_rate)}
    # Using the optimizer defined by optimizer_type
    optimizer = optimizer_collection[optimizer_type]
    # compute gradient
    gradients = optimizer.compute_gradients(loss)
    # apply gradient clipping to prevent gradient explosion
    capped_gradients = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gradients if grad is not None]
    # update the
    opt = optimizer.apply_gradients(capped_gradients, global_step=global_step)
    # opt = tf.train.AdamOptimizer(L).minimize(loss)
    summary_loss = tf.summary.scalar('loss', loss)

    true = tf.argmax(input_y, axis=1)
    pred_softmax = tf.nn.softmax(logits)
    pred = tf.argmax(pred_softmax, axis=1)
    correct_prediction = tf.equal(tf.cast(pred, tf.int64), true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summary_acc = tf.summary.scalar('acc', accuracy)
    precision, precision_op = tf.metrics.precision(true, pred,name="my_metric")
    recall, recall_op = tf.metrics.recall(true, pred,name="my_metric")
    tf_metric, tf_metric_update = tf.metrics.accuracy(true,
                                                      pred,
                                                      name="my_metric")
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
    # Define initializer to initialize/reset running variables
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)



# 运行计算图
with tf.Session(graph=graph) as sess:
    # define summary file writer
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(running_vars_initializer)
    # writer = tf.summary.FileWriter("./summary/visualization/", graph=graph)
    # merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=max_model_num)
    step = 0
    for epc in range(epoch):
        lo = 0.0
        for x1, y1, seq_length in get_batch(train_data, train_label, word2id["PAD"], batch_size):
            # print(12345)
            # print(999, x1.shape)
            _, loss_,pre,rec,pr_op,re_op,acc = sess.run([opt, loss, precision, recall, precision_op, recall_op, accuracy],
                                                        {input_x: x1,
                                                              input_y: y1,
                                                              L: lr,
                                                              sequence_length: seq_length,
                                                              })
            print("step", step,"精确率：", pre,"召回率：", rec)

            # loss.append(loss_)
            # writer.add_summary(s_m, global_step=step)
            lo = lo+loss_
            if step % 100 == 0:
                saver.save(sess, save_path="./ckpt/model.ckpt", global_step=step)
                # print("混淆矩阵：")
                # print(confusion)
            step += 1

        # print("训练Step:{:>5},loss:{:>7.4f},Accuracy:{:>7.2%}".format(step, lo, acc))
        # print("***",acc)
        # print("loss:",lo)


