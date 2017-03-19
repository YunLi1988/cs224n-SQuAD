from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
#import pgb

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.nn import dynamic_rnn, bidirectional_dynamic_rnn

from evaluate import exact_match_score, f1_score
from util import ConfusionMatrix, Progbar, minibatches, get_minibatches
from defs import LBLS

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn




class LSTMAttnCell(tf.nn.rnn_cell.LSTMCell):
    def _init_(self, num_units, encoder_output, scope=None):
        self.hs = encoder_output
        super(LSTMAttnCell,self).__init__(num_units,encoder_output, scope)
    
    def __call__(self, inputs, state, scope=None):
        lstm_out, lstm_state = super(LSTMAttnCell,self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn"):
                ht = tf.nn.rnn_cell._linear(lstm_out, self._num_units, True, 1.0)
                ht = tf.expand_dims(ht, axis=1)
            scores = tf.reduce_sum(self.hs*ht, reduction_indices=2, keep_dims=True)
            context = tf.reduce_sum(self.hs*scores, reduction_indices=1)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(tf.nn.rnn_cell._linear([context, lstm_out], self._num_units, True, 1.0))
        return (out, out)

class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def length(self, mask):
        used = tf.cast(mask, tf.int32)
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length


    def encode_questions(self, inputs, masks, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input with shape = (batch_size, length/max_length, embed_size)
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        if encoder_state_input == None:
            encoder_state_input = tf.zeros([1, self.size])
        cell_size = self.size
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        #initial_state_fw_cell = tf.slice(encoder_state_input, [0,0],[-1,cell_size])
        #initial_state_bw_cell = tf.slice(encoder_state_input, [0,cell_size],[-1,cell_size])
        #cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, state_is_tuple=True)
        #cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=cell_size, state_is_tuple=True)
        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.size)

        with tf.variable_scope("bi_LSTM"):
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(    
                                            cell_fw,
                                            cell_bw,
                                            dtype=tf.float32,
                                            sequence_length=self.length(masks),
                                            inputs= inputs,
                                            time_major = False
                                            )
    
        
        final_state = tf.concat(2, final_state)
        states = tf.concat(2, outputs)
        return final_state, states
    
    def encode_w_attn(self, inputs, masks, prev_states, scope="", reuse=False):
        """
        Run a BiLSTM over the context paragraph conditioned on the question representation.
        """
        self.attn_cell_fw = LSTMAttnCell(self.size, prev_states)
        self.attn_cell_bw = LSTMAttnCell(self.size, prev_states)
        with vs.variable_scope(scope, reuse):
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(    
                                            self.attn_cell_fw,
                                            self.attn_cell_bw,
                                            dtype=tf.float32,
                                            sequence_length=self.length(masks),
                                            inputs= inputs,
                                            time_major = False             
                                            )
    
        
        final_state = tf.concat(2, final_state)
        states = tf.concat(2, outputs)
        return final_state, states



class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def match_LASTM(self,questions_states, paragraph_states):
        input_size = tf.shape(questions_states)[2]
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.output_size, state_is_tuple=True)
        W_q = tf.get_variable("W_q", shape=(input_size, input_size), initializer=tf.contrib.layers.xavier_initializer())
        W_r = tf.get_variable("W_r", shape=(input_size, input_size), initializer=tf.contrib.layers.xavier_initializer())
        b_p = tf.get_variable("b_p", shape=(1,input_size), initializer=tf.contrib.layers.xavier_initializer())
        w = tf.get_variable("w", shape=(1, input_size), initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", shape=(1,1), initializer=tf.contrib.layers.xavier_initializer())
        state = tf.zeros([1, self.output_size])

        with tf.variable_scope("Forward_Match-LSTM"):
            for time_step in range(self.output_size):
                p_state = paragraph_states[:,time_step,:]
                G = tf.nn.tanh(tf.matmul(W_q, questions_states) + tf.mathmul(p_state,W_r) + tf.mathmul(state,W_r)+b_p)
                atten = tf.nn.softmax(tf.matmul(w, G) + b)
                z = tf.concat([p_state, tf.mathmul(questions_states,tf.transpose(atten))],1)
                state, h = cell(z, state, scope="Match-LSTM")
                tf.get_variable_scope().reuse_variables()
        fw_states = h

        W_q = tf.get_variable("W_q", shape=(input_size, input_size), initializer=tf.contrib.layers.xavier_initializer())
        W_r = tf.get_variable("W_r", shape=(input_size, input_size), initializer=tf.contrib.layers.xavier_initializer())
        b_p = tf.get_variable("b_p", shape=(1,input_size), initializer=tf.contrib.layers.xavier_initializer())
        w = tf.get_variable("w", shape=(1, input_size), initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", shape=(1,1), initializer=tf.contrib.layers.xavier_initializer())
        state = tf.zeros([1, self.output_size])
        with tf.variable_scope("Backward_Match-LSTM"):
            for time_step in reversed(range(self.output_size)):
                p_state = paragraph_states[:,time_step,:]
                G = tf.nn.tanh(tf.matmul(W_q, questions_states) + tf.mathmul(p_state,W_r) + tf.mathmul(state,W_r)+b_p)
                atten = tf.nn.softmax(tf.matmul(w, G) + b)
                z = tf.concat([p_state, tf.mathmul(questions_states,tf.transpose(atten))],1)
                state, h = cell(z, state, scope="Backward_Match-LSTM")
                tf.get_variable_scope().reuse_variables()  
        bk_states = h     
        knowledge_rep =  tf.concat(0,[fw_states,bk_states])
        return knowledge_rep


    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        input_size = tf.shape(knowledge_rep)[0]
        paragraph_len = tf.shape(knowledge_rep)[1]
        # predict start index
        cell = tf.nn.rnn_cell.LSTMCell(num_units=input_size, state_is_tuple=True)
        V = tf.get_variable("V", shape=(input_size/2, input_size), initializer=tf.contrib.layers.xavier_initializer())
        b_a = tf.get_variable("b_a", shape=(input_size/2,1), initializer=tf.contrib.layers.xavier_initializer())
        W_a = tf.get_variable("W_a", shape=(input_size/2, input_size/2), initializer=tf.contrib.layers.xavier_initializer())
        c = tf.get_variable("c", shape=(1,1), initializer=tf.contrib.layers.xavier_initializer())
        v = tf.get_variable("v", shape=(1,input_size/2), initializer=tf.contrib.layers.xavier_initializer())
        state = tf.zeros([input_size, 1])

        with tf.variable_scope("Boundary-LSTM_start"):
            for time_step in range(self.output_size):
                F_s = tf.nn.tanh(tf.matmul(V, knowledge_rep) + tf.mathmul(W_a,state) +b_a)
                beta_s = tf.nn.softmax(tf.matmul(v, F_s) + c)
                z = tf.mathmul(knowledge_rep,tf.transpose(beta))
                state, h = cell(z, state, scope="Boundary-LSTM_start")
                tf.get_variable_scope().reuse_variables()

        # predict end index; beta_e is the probability distribution over the paragraph words
        cell = tf.nn.rnn_cell.LSTMCell(num_units=input_size, state_is_tuple=True)
        V = tf.get_variable("V", shape=(input_size/2, input_size), initializer=tf.contrib.layers.xavier_initializer())
        b_a = tf.get_variable("b_a", shape=(input_size/2,1), initializer=tf.contrib.layers.xavier_initializer())
        W_a = tf.get_variable("W_a", shape=(input_size/2, input_size/2), initializer=tf.contrib.layers.xavier_initializer())
        c = tf.get_variable("c", shape=(1,1), initializer=tf.contrib.layers.xavier_initializer())
        v = tf.get_variable("v", shape=(1,input_size/2), initializer=tf.contrib.layers.xavier_initializer())
        state = tf.zeros([input_size, 1])

        with tf.variable_scope("Boundary-LSTM_end"):
            for time_step in range(self.output_size):
                F_e = tf.nn.tanh(tf.matmul(V, knowledge_rep) + tf.mathmul(W_a,state) +b_a)
                beta_e = tf.nn.softmax(tf.matmul(v, F_e) + c)
                z = tf.mathmul(knowledge_rep,tf.transpose(beta))
                state, h = cell(z, state, scope="Boundary-LSTM_")
                tf.get_variable_scope().reuse_variables()
        return beta_s, beta_e


class QASystem(object):
    def __init__(self, encoder, decoder, args, pretrained_embeddings):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.decoder = decoder
        self.config = args
        self.pretrained_embeddings = pretrained_embeddings
        # ==== set up placeholder tokens ========
        self.p_max_length = decoder.output_size
        self.embed_size = encoder.vocab_dim
        self.q_max_length = self.config.question_size
        self.q_placeholder = tf.placeholder(tf.int32, (None,self.q_max_length))
        self.p_placeholder = tf.placeholder(tf.int32, (None,self.p_max_length))    
        self.start_labels_placeholder = tf.placeholder(tf.int32, (None, self.p_max_length))
        self.end_labels_placeholder = tf.placeholder(tf.int32, (None, self.p_max_length))
        self.q_mask_placeholder = tf.placeholder(tf.bool, (None, self.q_max_length))
        self.p_mask_placeholder = tf.placeholder(tf.bool, (None, self.p_max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, ())

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.preds = self.decoder.decode(self.knowledge_rep)
            self.loss = self.setup_loss(self.preds)
        
        # ==== set up training/updating procedure ====
        


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        encoded_q, self.q_states= self.encoder.encode_questions(self.q_embeddings, self.q_mask_placeholder, None)
        encoded_p, self.p_states = self.encoder.encode_w_attn(self.p_embeddings, self.p_mask_placeholder, self.q_states, scope="", reuse=False)
        
        self.knowledge_rep = self.decoder.match_LASTM(self.q_states,self.p_states)

    def setup_loss(self, preds):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("start_index_loss"):  
            loss_tensor = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(preds[:,0], self.start_labels_placeholder),self.p_mask_placeholder)
            start_index_loss = tf.reduce_mean(loss_tensor, 0)
        with vs.variable_scope("end_index_loss"):  
            loss_tensor = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(preds[:,1], self.end_labels_placeholder),self.p_mask_placeholder)
            end_index_loss = tf.reduce_mean(loss_tensor, 0)
        self.loss = [start_index_loss, end_index_loss]

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            self.pretrained_embeddings = tf.Variable(self.pretrained_embeddings, trainable=False, dtype=tf.float32)
            q_embeddings = tf.nn.embedding_lookup(self.pretrained_embeddings, self.q_placeholder)
            self.q_embeddings = tf.reshape(q_embeddings, shape = [-1, self.config.question_size, 1* self.embed_size])
            p_embeddings = tf.nn.embedding_lookup(self.pretrained_embeddings, self.p_placeholder)
            self.p_embeddings = tf.reshape(p_embeddings, shape = [-1, self.config.output_size, 1* self.embed_size])

    def optimize(self, session, dataset, mask, dropout=1):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}
        if train_x is not None:
            input_feed[self.q_placeholder] = dataset['Questions']
            input_feed[self.p_placeholder] = dataset['Paragraphs']
        if train_y is not None:
            input_feed[self.start_labels_placeholder] = dataset['Labels'][:,0]
            input_feed[self.end_labels_placeholder] = dataset['Labels'][:,1]
        if mask is not None:
            input_feed[self.q_mask_placeholder] = dataset['Questions_masks']
            input_feed[self.p_mask_placeholder] = dataset['Paragraphs_masks']
        input_feed[self.dropout_placeholder] = dropout
        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        
        output_feed = []
        train_op_start = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.start_index_loss) 
        output_feed = [train_op_start, self.start_index_loss]
        start_index_pred = session.run(output_feed, input_feed)
        train_op_end = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.end_index_loss) 
        output_feed = [train_op_end, self.end_index_loss]
        end_index_pred = session.run(output_feed, input_feed)
        
        return start_index_loss, end_index_loss

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x
        #feed = self.create_feed_dict(inputs_batch)
        #predictions = sess.run(self.pred, feed_dict=feed)
        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, train_x, mask):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}
        if train_x is not None:
            input_feed[self.q_placeholder] = train_x['Questions']
            input_feed[self.p_placeholder] = train_x['Paragraphs']
        if mask is not None:
            input_feed[self.q_mask_placeholder] = train_x['Questions_masks']
            input_feed[self.p_mask_placeholder] = train_x['Paragraphs_masks']
        # fill in this feed_dictionary like:
        #input_feed['test_x'] = test_x
        
        output_feed = [self.preds]
        outputs = session.run(output_feed, input_feed)

        return outputs

    def run_epoch(self, sess, inputs, labels):
        """Runs an epoch of training.

        Args:
            sess: tf.Session() object
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch in get_minibatches([inputs, labels], self.config.batch_size):
            n_minibatches += 1
            total_loss += self.train_on_batch(sess, input_batch, labels_batch)
        return total_loss / n_minibatches

    def fit(self, sess, inputs, labels):
        """Fit model on provided data.

        Args:
            sess: tf.Session()
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            losses: list of loss per epoch
        """
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, inputs, labels)
            duration = time.time() - start_time
            print ('Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration))
            losses.append(average_loss)
        return losses
    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
            best_score = 0.

            train_examples = self.preprocess_sequence_data(train_examples_raw)
            dev_set = self.preprocess_sequence_data(dev_set_raw)

            for epoch in range(self.config.n_epochs):
                logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
                score = self.run_epoch(sess, train_examples, dev_set, train_examples_raw, dev_set_raw)
                if score > best_score:
                    best_score = score
                    if saver:
                        logger.info("New best score! Saving model in %s", self.config.model_output)
                        saver.save(sess, self.config.model_output)
                print("")
                if self.report:
                    self.report.log_epoch()
                    self.report.save()
            return best_score
    def run_epoch(self, sess, train_examples, dev_set, train_examples_raw, dev_set_raw):
        prog = Progbar(target=1 + int(len(train_examples['Paragraphs_masks']) / self.config.batch_size))
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            prog.update(i + 1, [("train loss", loss)])
            if self.report: self.report.log_train_loss(loss)
        print("")

        #logger.info("Evaluating on training data")
        #token_cm, entity_scores = self.evaluate(sess, train_examples, train_examples_raw)
        #logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        #logger.debug("Token-level scores:\n" + token_cm.summary())
        #logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        logger.info("Evaluating on development data")
        token_cm, entity_scores = self.evaluate(sess, dev_set, dev_set_raw)
        logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        logger.debug("Token-level scores:\n" + token_cm.summary())
        logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        f1 = entity_scores[-1]
        return f1
    def answer(self, session, test_x, mask):

        yp, yp2 = self.decode(session, test_x, mask)
        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)
        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        idx_sample = np.random.randint(0,dataset['Questions'].shape[0],sample)
        examples = {}
        examples['Questions'] = dataset['Questions'][idx_sample]
        examples['Paragraphs'] = dataset['Paragraphs'][idx_sample] 
        examples['Questions_masks'] = dataset['Questions'][idx_sample]
        examples['Paragraphs_masks'] = dataset['Paragraphs'][idx_sample]
        examples['Labels'] = dataset['Labels'][idx_sample] 
        
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for _, labels, labels_  in self.answer(sess, examples, masks):
            pred = set()
            if labels_[0] <= labels_[1]:
                pred = set(range(labels_[0],labels_[1]+1))
            gold = set(range(labels[0],labels[1]+1))
            
            correct_preds += len(gold.intersection(pred))
            total_preds += len(pred)
            total_correct += len(gold)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        em = correct_preds

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
 
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        


