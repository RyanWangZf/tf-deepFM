# -*- coding: utf-8 -*-
import tensorflow as tf
import pdb
import sys
import numpy as np 
from sklearn.metrics import roc_auc_score

from config import config

class deepFM(object):
    def __init__(self,config):
        self.config = config
        # build session
        if config.use_gpu:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
        else:
            sess_config = tf.ConfigProto(device_count = {"CPU": 4},
                    inter_op_parallelism_threads=0,
                    intra_op_parallelism_threads=0)
        self.sess = tf.Session(config=sess_config)

    def train(self,dataset):
        config = self.config
        # start training
        sample_total_num = dataset.all_data[0]["feature"].shape[0]
        num_iter_one_epoch = sample_total_num // config.batch_size
        feat_tensor, value_tensor, label_tensor = dataset.get_batch()

        # build graph
        pred = self._inference(feat_tensor, value_tensor)
        loss, logloss = self._loss_func(pred,label_tensor)
        train_op = self._optimizer(loss)

        self.sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        saver = tf.train.Saver(max_to_keep=2)

        for step in range(config.num_epoch):
            # init training dataset
            dataset.init_iterator(self.sess,is_training=True)

            for iteration in range(num_iter_one_epoch):
                batch_logloss,_ = \
                     self.sess.run([logloss,train_op])

                sys.stdout.write("\r")
                sys.stdout.write("=> [INFO] Process {:.0%} in Epoch {:d}: [Train] log-loss: {:.5f} <= \r".format(iteration/num_iter_one_epoch, step+1,batch_logloss))
                sys.stdout.flush()

            if dataset.va_filename is not None:
                print("\n")
                # init va dataset
                dataset.init_iterator(self.sess,is_training=False)
                va_epoch_loss = 0
                val_count = 0
                va_pred = []
                try:
                    while True:
                        va_b_epoch_loss,va_b_pred = self.sess.run([logloss,pred])
                        va_epoch_loss += va_b_epoch_loss
                        val_count += 1
                        va_pred.extend(va_b_pred.tolist())

                except tf.errors.OutOfRangeError:
                    va_pred = np.array(va_pred)
                    val_auc = roc_auc_score(dataset.all_data[1]["label"],va_pred)
                    print("=> [INFO] Epoch {}, [Val] val_loss: {:.5f}, val_auc: {:.3f} <=".format(step+1,va_epoch_loss/val_count,val_auc))
            
            # save model
            saver.save(self.sess,config.model_dir+"/deepFM.ckpt",global_step=step+1)

        return

    def _optimizer(self,loss):
        config = self.config
        # build optimizer
        opt = tf.train.AdamOptimizer(config.learning_rate)

        # build train op
        params = tf.trainable_variables()
        gradients = tf.gradients(loss,params,colocate_gradients_with_ops=True)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
        train_op = opt.apply_gradients(zip(clipped_grads, params))
        return train_op

    def _loss_func(self,pred,label):
        config = self.config

        with tf.name_scope("l2_loss"):
            # tf.get_collection(tf.GraphKeys.WEIGHTS)
            # l2 normalization
            regularizer = tf.contrib.layers.l2_regularizer(config.l2_norm)
            reg_term = tf.contrib.layers.apply_regularization(regularizer,weights_list=None)

        with tf.name_scope("logistic_loss"):
            # logistic loss
            logit_1 = tf.log(pred + 1e-10)
            logit_0 = tf.log(1 - pred + 1e-10)
            log_loss = -1 * tf.reduce_mean(label * logit_1 + (1- label) * logit_0)  

        total_loss = log_loss + reg_term

        return total_loss,log_loss

    def _inference(self,feature_tensor,value_tensor):
        config = self.config

        # factorization machine
        with tf.variable_scope("fm"):
            # linear part
            mask = tf.expand_dims(value_tensor, 2) # None,m,1

            bias = tf.get_variable("bias", shape=[1],
                        initializer = tf.zeros_initializer())
            weight = tf.get_variable("weight", shape=[config.n,1],
                        initializer = tf.truncated_normal_initializer(mean=0,stddev=1e-2))

            linear_term = tf.multiply(tf.gather(weight,feature_tensor), mask) # None,m,1
            linear_term = tf.add(bias, tf.reduce_sum(linear_term,[-1,-2])) # None,

            # interaction part
            fm_v = tf.get_variable("v", shape=[config.n, config.k],
                    initializer = tf.truncated_normal_initializer(mean=0,stddev=1e-2))

            vx_embedding = tf.gather(fm_v, feature_tensor) # None,m,k
            vx_embedding = tf.multiply(vx_embedding, mask) # None,m,k

            vx2 = tf.square(tf.reduce_sum(vx_embedding,1)) # None,k
            v2x2 = tf.square(vx_embedding) # None,m,k
            v2x2 = tf.reduce_sum(v2x2,1) # None,k

            fm_logit = 0.5 * tf.reduce_sum(vx2 - v2x2, 1) # None,

            tf.add_to_collection(tf.GraphKeys.WEIGHTS,weight)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS,bias)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS,fm_v)

        # deep components
        with tf.variable_scope("deep"):
            # embedding layer
            dnn_embed = tf.gather(fm_v, feature_tensor) # None,m,k
            # concat the embeddings
            dnn_hidden = tf.reshape(dnn_embed, [-1, config.m * config.k]) # None,m*k

            # hidden layers
            for idx, units in enumerate(config.hidden_units):
                dnn_hidden = dense(dnn_hidden, units, "relu", "dnn_{}".format(idx))

            # output layer
            deep_logit = tf.reshape(dense(dnn_hidden,1,None,"dnn_output"),[-1,]) # None,

        logit = tf.add(fm_logit, deep_logit)
        prob = tf.sigmoid(logit)

        return prob

def dense(h,n_unit,activation="relu",scope=None):
    """A customized dense layer for deep fm
    """
    input_dim = h.get_shape().as_list()[1]

    with tf.variable_scope(scope):
        w = tf.get_variable("w", shape=[input_dim, n_unit],
            initializer = tf.truncated_normal_initializer(mean=0, stddev=1e-2))
        b = tf.get_variable("b", shape=[n_unit],
            initializer = tf.zeros_initializer())
        h = tf.matmul(h, w) + b
        if activation is not None:
            # TODO only support relu currently
            h = tf.nn.relu(h)

    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,b)
    return h

def train_deepfm():
    from config import config
    from dataset import Dataset

    deepfm = deepFM(config)
    tr_filename = "data/criteo.tr.r100.gbdt0.ffm"
    va_filename = "data/criteo.va.r100.gbdt0.ffm"
    dataset = Dataset(tr_filename,va_filename,config.batch_size,config.shuffle)
    deepfm.train(dataset)
    return


if __name__ == '__main__':
    train_deepfm()