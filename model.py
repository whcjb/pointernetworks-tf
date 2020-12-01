import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from layers import *

class Model(object):
  def __init__(self, config, 
               inputs, labels, enc_seq_length, dec_seq_length, mask,
               reuse=False, is_critic=False):
    self.task = config.task
    self.debug = config.debug
    self.config = config

    self.input_dim = config.input_dim
    self.hidden_dim = config.hidden_dim
    self.num_layers = config.num_layers

    self.max_enc_length = config.max_enc_length
    self.max_dec_length = config.max_dec_length
    self.num_glimpse = config.num_glimpse

    self.init_min_val = config.init_min_val
    self.init_max_val = config.init_max_val
    self.initializer = \
        tf.random_uniform_initializer(self.init_min_val, self.init_max_val)

    self.use_terminal_symbol = config.use_terminal_symbol # True

    self.lr_start = config.lr_start # 0.001
    self.lr_decay_step = config.lr_decay_step # 5000
    self.lr_decay_rate = config.lr_decay_rate # 0.96
    self.max_grad_norm = config.max_grad_norm # 2.0

    self.layer_dict = {}

    ##############
    # inputs
    ##############

    self.is_training = tf.placeholder_with_default(
        tf.constant(False, dtype=tf.bool),
        shape=(), name='is_training'
    )

    self.enc_inputs, self.dec_targets, self.enc_seq_length, self.dec_seq_length, self.mask = \
        smart_cond(
            self.is_training,
            lambda: (inputs['train'], labels['train'], enc_seq_length['train'],
                     dec_seq_length['train'], mask['train']),
            lambda: (inputs['test'], labels['test'], enc_seq_length['test'],
                     dec_seq_length['test'], mask['test'])
        )
    print('self.enc_inputs: ', self.enc_inputs.shape) # (128, 10, 2)
    print('self.dec_targets: ', self.dec_targets.shape) # (128, 10)
    print('self.enc_seq_length: ', self.enc_seq_length.shape) #(128,) 
    print('self.dec_seq_length: ', self.dec_seq_length.shape) #(128,)
    print('self.mask: ', self.mask.shape) # (128, 11)

    #self.dec_seq_length = tf.Print(self.dec_seq_length, [self.dec_seq_length], 'Print dec_seq_length') # [10, 10, 10 ...] * 128

    if self.use_terminal_symbol:
      print('self.dec_seq_length1: ', self.dec_seq_length) # (128,)
      # self.dec_seq_length = tf.Print(self.dec_seq_length, [self.dec_seq_length], 'self.dec_seq_length1: ') # [10 10 10...]
      self.dec_seq_length += 1 # terminal symbol
      # self.dec_seq_length = tf.Print(self.dec_seq_length, [self.dec_seq_length], 'self.dec_seq_length2: ') # [11 11 11...]

      print('self.dec_seq_length2: ', self.dec_seq_length) # (128,)

    self._build_model()
    self._build_steps()

    if not reuse:
      self._build_optim()

    self.train_summary = tf.summary.merge([
        tf.summary.scalar("train/total_loss", self.total_loss),
        tf.summary.scalar("train/lr", self.lr),
    ])

    self.test_summary = tf.summary.merge([
        tf.summary.scalar("test/total_loss", self.total_loss),
    ])

  def _build_steps(self):
    def run(sess, fetch, feed_dict, summary_writer, summary):
      fetch['step'] = self.global_step
      if summary is not None:
        fetch['summary'] = summary

      result = sess.run(fetch)
      if summary_writer is not None:
        summary_writer.add_summary(result['summary'], result['step'])
        summary_writer.flush()
      return result

    def train(sess, fetch, summary_writer):
      return run(sess, fetch, feed_dict={},
                 summary_writer=summary_writer, summary=self.train_summary)

    def test(sess, fetch, summary_writer=None):
      return run(sess, fetch, feed_dict={self.is_training: False},
                 summary_writer=summary_writer, summary=self.test_summary)

    self.train = train
    self.test = test
# ===============================model============
  def _build_model(self):
    tf.logging.info("Create a model..")
    self.global_step = tf.Variable(0, trainable=False)

#---------------------------------------------------------- embed
    input_embed = tf.get_variable( # tf.get_variable(name,  shape, initializer): name就是变量的名称，shape是变量的维度，initializer是变量初始化的方式
        "input_embed", [1, self.input_dim, self.hidden_dim], # [1, 2, 256]
        initializer=self.initializer)

    with tf.variable_scope("encoder"):
      self.embeded_enc_inputs = tf.nn.conv1d( # (values, filters)
          # (128, 10, 2) * [1, 2, 256] -> (128, 10, 256)
          self.enc_inputs, input_embed, 1, "VALID") # [[bs, 10, 2]
      print('self.embeded_enc_inputs: ', self.embeded_enc_inputs) # (128, 10, 256)

#---------------------------------------------------------- encoder
    batch_size = tf.shape(self.enc_inputs)[0] # 128
    #batch_size = tf.Print(batch_size, [self.num_layers], 'num_layers: ') # [1]
    with tf.variable_scope("encoder"):
      self.enc_cell = LSTMCell(
          self.hidden_dim, # num_units, 网络单元的个数，即隐藏层的节点数
          initializer=self.initializer)


      if self.num_layers > 1: # 1
        cells = [self.enc_cell] * self.num_layers # [5] * 5 => [5, 5, 5, 5, 5]
        self.enc_cell = MultiRNNCell(cells) # MultiRNNCell([list RNNcell], state_is_tuple=True)
      
      # hidden_states的初始值，输入到rnn中
      # pytorch内部自动初始化，先不考虑
      self.enc_init_state = trainable_initial_state(
          batch_size, self.enc_cell.state_size)

      #不用考虑initial_state，pytorch内部会自动初始化self.embeded_enc_inputs = tf.Print(self.embeded_enc_inputs, [tf.shape(self.enc_init_state)], 'self.enc_init_state: ') # [2 16 256]


      # self.encoder_outputs : [None, max_time, output_size]
      # https://zhuanlan.zhihu.com/p/43041436
      # enc_outputs(128, 10, 256), enc_final_states(128, 256)
      self.enc_outputs, self.enc_final_states = \
                tf.nn.dynamic_rnn(self.enc_cell,  # encoder总体
                                  self.embeded_enc_inputs,
                                  self.enc_seq_length, 
                                  self.enc_init_state) # rnn cell 初始值输入

      # 初始值是[0, 0, ...], 会梯度更新值
      self.first_decoder_input = tf.expand_dims(trainable_initial_state(
          batch_size, self.hidden_dim, name="first_decoder_input"), 1) # (128, 1, 256)

      if self.use_terminal_symbol: 
        # 0 index indicates terminal
        self.enc_outputs = tf.concat( # 城市index是从1开始
            [self.first_decoder_input, self.enc_outputs], axis=1) # (128, 11, 256)

#---------------------------------------------------------- decoder train
    with tf.variable_scope("decoder"):
      # 把city排序加上index, 共有128个，比如
      # [[[0,2],[0,3],[0,5],[0,1],[0,4]],
      #  [[1,3],[1,2],[1,5],[1,4],[1,1]],
      #  ...
      #  [[127,5],[127,1],[127,3],[127,4],[127,2]]
      #  ]
      self.idx_pairs = index_matrix_to_pairs(self.dec_targets) # (128, 10) -> (128, 10, 2)

      # stop_gradient即这个部分不需要计算梯度。
      # enc_outputs (128, 11, 256), (128, 10, 2)
      self.embeded_dec_inputs = tf.stop_gradient(
          tf.gather_nd(self.enc_outputs, self.idx_pairs)) # 根据城市位置调整输入特征 # (128, 10, 256)

      if self.use_terminal_symbol: # 使用0最为结束符号, 在target最后一位设置0
        tiled_zero_idxs = tf.tile(tf.zeros(
            [1, 1], dtype=tf.int32), [batch_size, 1], name="tiled_zero_idxs")
        # [[0]] -> [[0],[0]...[0]], 128个
        self.dec_targets = tf.concat([self.dec_targets, tiled_zero_idxs], axis=1) # (128, 11) [[2,1,4,5,6,7,0],[5,4,3,2,1,6,7,0],...[]]
        # [1 5 4 9 6 10 7 2 3 8 0][1 2 7 5 8 4 6 9 10 3 0][1 8 9 2 3 7 4 10 5 6 0][1 2 4 10 8 5 3 9 6 7 0][1 7 3 10 5 2 6 8 4 9 0][1 4 5 10 3 2 6 8 9 7 0][1 4 7 9 8 6 3 2 10 5 0][1 6 5 7 8 2 9 4 10 3 0][1 7 3 10 6 5 8 2 9 4 0][1 6 8 10 2 9 5 3 4 7 0][1 6 4 8 5 3 2 9 10 7 0]
        #self.dec_targets = tf.Print(self.dec_targets, [self.dec_targets], 'self.dec_targets: ', summarize=200)

      self.embeded_dec_inputs = tf.concat(
          # cat([128, 1, 256], [128, 10, 256]) 
          [self.first_decoder_input, self.embeded_dec_inputs], axis=1) # (128, 11, 256)

      self.dec_cell = LSTMCell(
          self.hidden_dim,
          initializer=self.initializer)

      if self.num_layers > 1:
        cells = [self.dec_cell] * self.num_layers
        self.dec_cell = MultiRNNCell(cells)

      self.dec_pred_logits, _, _ = decoder_rnn( # train
          self.dec_cell, 
          self.embeded_dec_inputs, 
          self.enc_outputs, 
          self.enc_final_states,
          self.dec_seq_length, 
          self.hidden_dim,
          self.num_glimpse, 
          batch_size, 
          is_train=True,
          initializer=self.initializer)

      self.dec_pred_prob = tf.nn.softmax(
          self.dec_pred_logits, 2, name="dec_pred_prob")
      self.dec_pred = tf.argmax(
          self.dec_pred_logits, 2, name="dec_pred")
#----------------------------------------------------------infrence
    with tf.variable_scope("decoder", reuse=True):
      self.dec_inference_logits, _, _ = decoder_rnn( # inference # (128, ?, 11)
          self.dec_cell, 
          self.first_decoder_input,
          self.enc_outputs, 
          self.enc_final_states,
          self.dec_seq_length, 
          self.hidden_dim,
          self.num_glimpse, 
          batch_size, 
          is_train=False,
          initializer=self.initializer,
          max_length=self.max_dec_length + int(self.use_terminal_symbol))

      self.dec_inference_prob = tf.nn.softmax(
          self.dec_inference_logits, 2, name="dec_inference_logits") # (128, ?, 11)
      self.dec_inference = tf.argmax(
          self.dec_inference_logits, 2, name="dec_inference") # (128, ?)

  def _build_optim(self):
    #self.dec_targets = tf.Print(self.dec_targets, [tf.shape(self.dec_targets)], 'self.dec_targets: ', summarize=11) # [16 11 11]
    # dec_pred_logits正负值都有，但是没有归一化
    #self.dec_targets = tf.Print(self.dec_targets, [self.dec_pred_logits[0, :]], 'self.dec_pred_logits: ', summarize=11) # [16 11]
    
    #self.dec_targets = tf.Print(self.dec_targets, [self.dec_targets[0, :]], 'self.dec_targets: ', summarize=11) # [1 7 6 4 8 3 2 10 9 5 0]
    # (128, 11)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.dec_targets, logits=self.dec_pred_logits)

    inference_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.dec_targets, logits=self.dec_inference_logits)

    def apply_mask(op):
      length = tf.cast(op[:1], tf.int32)
      loss = op[1:]
      return tf.multiply(loss, tf.ones(length, dtype=tf.float32))

    batch_loss = tf.div(
        tf.reduce_sum(tf.multiply(losses, self.mask)),
        tf.reduce_sum(self.mask), name="batch_loss")

    batch_inference_loss = tf.div(
        tf.reduce_sum(tf.multiply(losses, self.mask)),
        tf.reduce_sum(self.mask), name="batch_inference_loss")

    tf.losses.add_loss(batch_loss)
    total_loss = tf.losses.get_total_loss()

    self.total_loss = total_loss
    self.target_cross_entropy_losses = losses
    self.total_inference_loss = batch_inference_loss

    self.lr = tf.train.exponential_decay(
        self.lr_start, self.global_step, self.lr_decay_step,
        self.lr_decay_rate, staircase=True, name="learning_rate")

      
    optimizer = tf.train.AdamOptimizer(self.lr)

    if self.max_grad_norm != None:
      grads_and_vars = optimizer.compute_gradients(self.total_loss)
      for idx, (grad, var) in enumerate(grads_and_vars):
        if grad is not None:
          grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
      self.optim = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
    else:
      self.optim = optimizer.minimize(self.total_loss, global_step=self.global_step)
