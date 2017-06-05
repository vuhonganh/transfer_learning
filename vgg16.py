########################################################################################
# Hong Anh VU, 2016                                                                    #
# VGG16 transfer learning implementation in TensorFlow                                 #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Conv Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from data_reader import get_data, get_data_stratify
import os
import sys


class vgg16:
    def __init__(self, weights=None, sess=None, learning_rate=0.01, fc_lay_weights=None):
        self.learning_rate = learning_rate
        self.fc3l = None
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.fc_parameters = []
        self._create_placeholder()
        self.convlayers()
        self.fc_layers_transfer()
        self._add_loss_op()
        self._add_train_op()
        self.test_probs = tf.nn.softmax(self.fc3l)
        self.summary_op = tf.summary.scalar("loss", self.loss)
        sess.run(tf.global_variables_initializer())
        if weights is not None and sess is not None:
            self.load_conv_weights(weights, sess)
        if fc_lay_weights is not None:
            self.load_fc_weights(fc_lay_weights, sess)


    def _create_placeholder(self):
        self.input_placeholder = tf.placeholder(tf.float32, [None, 224, 224, 3], "input")
        self.label_placeholder = tf.placeholder(tf.float32, [None, 12], "label")
        self.dropout_placeholder = tf.placeholder(tf.float32, [])

    def create_feed_dict(self, input_batch, label_batch=None, dropout=1.0):
        feed_dict = {}
        feed_dict[self.input_placeholder] = input_batch
        feed_dict[self.dropout_placeholder] = dropout
        if label_batch is not None:
            feed_dict[self.label_placeholder] = label_batch

        return feed_dict



    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images_avg = self.input_placeholder - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),
                                 name='weights',
                                 trainable=False)

            conv = tf.nn.conv2d(images_avg, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False,
                                 name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv1_1 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1),
                                 name='weights',
                                 trainable=False)

            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False,
                                 name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv1_2 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1),
                                 name='weights',
                                 trainable=False)

            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False,
                                 name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1),
                                 trainable=False,
                                 name='weights')

            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False,
                                 name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1),
                                 trainable=False,
                                 name='weights')

            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False,
                                 name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                                 trainable=False,
                                 name='weights')

            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False,
                                 name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                                 trainable=False,
                                 name='weights')

            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False,
                                 name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv3_3 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1),
                                 trainable=False,
                                 name='weights')

            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False,
                                 name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv4_1 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                                 trainable=False,
                                 name='weights')

            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False,
                                 name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv4_2 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                                 trainable=False,
                                 name='weights')

            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False,
                                 name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1),
                                 trainable=False,
                                 name='weights')

            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False,
                                 name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv5_1 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                                 trainable=False,
                                 name='weights')

            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False,
                                 name='biases')

            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                                 trainable=False,
                                 name='weights')

            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False,
                                 name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv5_3 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=False, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=False, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=False, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]


    def fc_layers_transfer(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            print(shape)
            fc1w = tf.Variable(tf.truncated_normal([shape, 1024],
                                                   dtype=tf.float32,
                                                   stddev=1.0/(1024 ** 0.5)),
                               trainable=True,
                               name='w1')

            fc1b = tf.Variable(tf.zeros(shape=[1024], dtype=tf.float32),
                               trainable=True,
                               name='b1')

            pool5_flat = tf.reshape(self.pool5, [-1, shape])

            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            fc1_rl = tf.nn.relu(fc1l)
            self.fc1 = tf.nn.dropout(fc1_rl, self.dropout_placeholder)

            self.fc_parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([1024, 256],
                                                   dtype=tf.float32,
                                                   stddev=1.0/(256 ** 0.5)),
                               trainable=True,
                               name='w2')

            fc2b = tf.Variable(tf.zeros(shape=[256], dtype=tf.float32),
                               trainable=True,
                               name='b2')

            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)

            fc2_rl = tf.nn.relu(fc2l)
            self.fc2 = tf.nn.dropout(fc2_rl, self.dropout_placeholder)
            self.fc_parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([256, 12], dtype=tf.float32, stddev=1.0/(12 ** 0.5)),
                               trainable=True,
                               name='w3')

            fc3b = tf.Variable(tf.zeros(shape=[12], dtype=tf.float32),
                               trainable=True,
                               name='b3')

            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)

            self.fc_parameters += [fc3w, fc3b]

    def _add_loss_op(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_placeholder,
                                                                           logits=self.fc3l))

    def _add_train_op(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                            global_step=self.global_step)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def load_conv_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())        
        for i, k in enumerate(keys[:-6]):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def load_fc_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = weights.keys()
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.fc_parameters[i].assign(weights[k]))


if __name__ == '__main__':
    num_train_step = 3200  # roughly 20 epoch
    # num_train_step = 2  # test purpose
    print_val_size = 10
    save_param_size = 160
    batch_size = 64
    continue_training = False

    learning_rate = 0.00002

    # images, labels, train_idx, val_idx, test_idx = get_data()
    images, labels, train_idx, val_idx, test_idx = get_data_stratify()
    length_train = len(train_idx)
    length_val = len(val_idx)
    length_test = len(test_idx)

    checkpoint_dir = './.checkpoints/'

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if continue_training:
            vgg = vgg16('vgg16_weights.npz', sess, learning_rate, 'fc_lay.npz')
        else:
            vgg = vgg16('vgg16_weights.npz', sess, learning_rate)

        # saver = tf.train.Saver(vgg.fc_parameters)

        # ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_dir))
        # if ckpt and ckpt.model_checkpoint_path:
        #     print("restoring model...")
        #     saver.restore(sess, ckpt.model_checkpoint_path)

        writer = tf.summary.FileWriter('./.log/', sess.graph)

        cur_id = vgg.global_step.eval()
        cur_val_id = 0
        for i in range(num_train_step):
            if cur_id > length_train:
                cur_id = 0
            idx_batch = train_idx[cur_id: cur_id+batch_size]
            cur_id += batch_size

            if len(idx_batch) == 0:
                continue

            input_images = images[idx_batch]
            input_labels = labels[idx_batch]
            feed = vgg.create_feed_dict(input_batch=input_images, label_batch=input_labels, dropout=0.5)

            _, loss_batch, summary = sess.run(fetches=[vgg.train_op, vgg.loss, vgg.summary_op],
                                              feed_dict=feed)
            writer.add_summary(summary, global_step=i)

            print("step %d: %f" % (i, loss_batch))

            if (i+1) % print_val_size == 0:
                idx_val_batch = val_idx[cur_val_id:cur_val_id+batch_size]
                cur_val_id += batch_size
                if cur_val_id > length_val:
                    cur_val_id = 0

                if len(idx_val_batch) == 0:
                    continue

                val_images = images[idx_val_batch]
                val_labels = labels[idx_val_batch]
                val_feed = vgg.create_feed_dict(input_batch=val_images)
                val_probs = sess.run(fetches=vgg.test_probs, feed_dict=val_feed)
                cnt = 0
                for id_val in range(len(val_labels)):
                    correct_idx = np.argmax(val_labels[id_val])
                    guess_idx = np.argmax(val_probs[id_val])
                    if correct_idx == guess_idx:
                        cnt += 1
                print("batch validation accuracy: %f" % (cnt / batch_size))

            if (i+1) % save_param_size == 0:
                print("saving to npz file...")
                # saver.save(sess, checkpoint_dir + 'model_', int((i+1)/skip_size))
                fc_layers = sess.run(fetches=vgg.fc_parameters)
                np.savez("fc_lay.npz", w1=fc_layers[0], b1=fc_layers[1], w2=fc_layers[2],
                         b2=fc_layers[3], w3=fc_layers[4], b3=fc_layers[5])
                print("saved to npz file!")

        print("final test:")
        cur_test_id = 0
        cnt = 0

        assignments = np.zeros((12, 12), dtype=np.float32)

        while cur_test_id < length_test:
            idx_test_batch = test_idx[cur_test_id: cur_test_id + batch_size]
            cur_test_id += batch_size
            test_images = images[idx_test_batch]
            test_labels = labels[idx_test_batch]
            test_feed = vgg.create_feed_dict(input_batch=test_images)
            test_probs = sess.run(fetches=vgg.test_probs, feed_dict=test_feed)
            for id_test in range(len(test_labels)):
                correct_idx = np.argmax(test_labels[id_test])
                guess_idx = np.argmax(test_probs[id_test])

                # table that shows which one is mapped to which one
                # correct idx is presented vertically on the left side
                assignments[correct_idx, guess_idx] += 1.0

                if correct_idx == guess_idx:
                    cnt += 1

            percentage = min(cur_test_id / length_test, 1.0)
            progress = '{0:.0%}'.format(percentage)
            print("progress done %s" % progress)

        print("test accuracy is %f" % (cnt / length_test))

        assignments = np.around(assignments / np.sum(assignments, axis=1)[:, None], 2)
        print("saving assignments table...")
        np.save("assignments_table", assignments, allow_pickle=False)
        print("assignments table is saved!")

