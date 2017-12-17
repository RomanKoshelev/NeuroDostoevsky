import os
from utils import make_dir
import numpy as np
import tensorflow as tf
import pickle
from visualization import show_train_stats

class WordRNN:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        # state
        self.tr_step   = 0
        self.tr_epoch  = 0
        self.tr_losses = []
        
    
    def _make_rnn_cell(self, num_units, num_layers, keep_prob):
        def make_layer():
            l = tf.contrib.rnn.BasicLSTMCell(num_units)
            l = tf.contrib.rnn.DropoutWrapper(l, output_keep_prob=keep_prob)
            return l        
        layers = [make_layer() for _ in range(num_layers)]
        cell   = tf.contrib.rnn.MultiRNNCell(layers)
        return cell

    
    def _make_loss(self, logits, targets, lstm_size, num_classes):
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, 
            labels=tf.one_hot(targets, num_classes))
        loss = tf.reduce_mean(loss)
        return loss
    

    def _make_optimizer(self, loss, lr, grad_clip):
        tr_vars   = tf.trainable_variables()
        grads, _  = tf.clip_by_global_norm(tf.gradients(loss, tr_vars), grad_clip)
        train_op  = tf.train.AdamOptimizer(lr)
        optimizer = train_op.apply_gradients(zip(grads, tr_vars))
        return optimizer
    
    
    def _get_initial_state(self, batch_size):
        return self._sess.run(self.initial_state, feed_dict={self.batch_size_pl: batch_size})

    
    def build(self, emb_size, num_units, num_layers, grad_clip):
        tf.reset_default_graph()
        self._graph = tf.Graph()
        self._scope  = "char_rnn"
        with self._graph.as_default(), tf.variable_scope(self._scope):
            
            # placeholders
            self.inputs_pl     = tf.placeholder(tf.int32, [None, None], name='inputs')
            self.targets_pl    = tf.placeholder(tf.int32, [None, None], name='targets')
            self.seq_length_pl = tf.placeholder(tf.int32, [None], 'seq_lengths')
            self.batch_size_pl = tf.placeholder(tf.int32, shape=[], name='batch_size')
            self.keep_prob_pl  = tf.placeholder(tf.float32, name='keep_prob')
            self.lr_pl         = tf.placeholder(tf.float32, name='learning_rate')

            # embedding
            embedding_mtx      = tf.Variable(tf.random_normal(shape=[self.num_classes, emb_size], dtype=tf.float32))
            embed              = tf.nn.embedding_lookup(embedding_mtx, self.inputs_pl)
            
            # network
            cell               = self._make_rnn_cell(num_units, num_layers, self.keep_prob_pl)
            initial_state      = cell.zero_state(self.batch_size_pl, tf.float32)
            outputs, state     = tf.nn.dynamic_rnn(cell, embed, self.seq_length_pl, initial_state, dtype=tf.float32)
            self.initial_state = initial_state
            self.final_state   = state

            # prediction
            logits             = tf.layers.dense(outputs, self.num_classes)
            self.prediction    = tf.nn.softmax(logits, name='predictions')

            # training
            self.loss_op       = self._make_loss(logits, self.targets_pl, num_units, self.num_classes)
            self.train_op      = self._make_optimizer(self.loss_op, self.lr_pl, grad_clip)
            
            # utils
            self.init_op       = tf.global_variables_initializer()
            self._saver        = tf.train.Saver()

        # session
        self._sess = tf.Session(graph=self._graph)
        self._sess.run(self.init_op)
        
        
    def train(self, dataset, seq_length, epochs, batch_size, keep_prob, learning_rate, log_every=20, mean_win=30):
        try:
            for self.tr_epoch in range(self.tr_epoch, epochs):
                state = self._get_initial_state(batch_size)

                for x, y in dataset.get_batches(batch_size, seq_length):
                    self.tr_step += 1
                    tr_loss, state, _ = self._sess.run(
                        [self.loss_op, self.final_state, self.train_op], 
                        feed_dict = {
                            self.inputs_pl    : x,
                            self.targets_pl   : y,
                            self.seq_length_pl: [seq_length, ]*batch_size,
                            self.initial_state: state,
                            self.keep_prob_pl : keep_prob,
                            self.lr_pl        : learning_rate,
                    })
                    self.tr_losses.append(tr_loss)
                    
                    if self.tr_step % log_every == 0:
                        show_train_stats(self.tr_epoch, self.tr_step, self.tr_losses, mean_win)
                        
        except KeyboardInterrupt:
            show_train_stats(self.tr_epoch, self.tr_step, self.tr_losses, mean_win)
            

    def save(self, path):
        make_dir(path)
        pickle.dump([self.tr_epoch, self.tr_step, self.tr_losses], open(os.path.join(path, "state.p"), "wb"))
        self._saver.save(self._sess, path)
        
    def restore(self, path):
        try:
             [self.tr_epoch, self.tr_step, self.tr_losses] = pickle.load(open(os.path.join(path, "state.p"), "rb"))
        except: 
            print("State not found at", path)
        self._saver.restore(self._sess, path)
        
    def predict(self, x, state, seq_len):
        pred, state = self._sess.run(
            [self.prediction, self.final_state], 
            feed_dict={
                self.inputs_pl    : x,
                self.seq_length_pl: [seq_len],
                self.initial_state: state,
                self.keep_prob_pl : 1.
            })
        return pred, state
    
    def sample(self, dataset, n_samples, top_n, prime):
        seq_len = 1
        def pick_top_n(pred, vocab_size, top_n):
            p = np.squeeze(pred)
            p[np.argsort(p)[:-top_n]] = 0
            p = p / np.sum(p)
            t = np.random.choice(vocab_size, 1, p=p)[0]
            return t

        # todo: tokenize instead of splitting
        prime   = prime.split(' ')
        samples = [w for w in prime]
        state = self._get_initial_state(batch_size = 1)

        # todo: use <unk> token when encode
        for w in prime: 
            x      = np.zeros([1, seq_len])
            x[0,0] = dataset.word_to_token[w]
            preds, state = self.predict(x, state, seq_len)
        t = pick_top_n(preds, self.num_classes, top_n)
        samples.append(dataset.token_to_word[t])

        for i in range(n_samples):
            x[0,0] = t
            preds, state = self.predict(x, state, seq_len)
            t = pick_top_n(preds, self.num_classes, top_n)
            samples.append(dataset.token_to_word[t])

        return ' '.join(samples)