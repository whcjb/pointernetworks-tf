import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import seq2seq
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

try:
  from tensorflow.contrib.layers.python.layers import utils
except:
  from tensorflow.contrib.layers import utils

smart_cond = utils.smart_cond

try:
  LSTMCell = rnn.LSTMCell
  MultiRNNCell = rnn.MultiRNNCell
  # dynamic_rnn_decoder = seq2seq.dynamic_rnn_decoder 
  # simple_decoder_fn_train = seq2seq.simple_decoder_fn_train
except:
  LSTMCell = tf.contrib.rnn.LSTMCell
  MultiRNNCell = tf.contrib.rnn.MultiRNNCell
  # dynamic_rnn_decoder = tf.contrib.legacy_seq2seq.dynamic_rnn_decoder
  # simple_decoder_fn_train = tf.contrib.legacy_seq2seq.simple_decoder_fn_train

# try:
#   from tensorflow.python.ops.gen_array_ops import _concat_v2 as concat_v2
# except:
#   concat_v2 = tf.concat_v2

# https://gist.github.com/alrojo/d66240bdb8fdb2658081c3918ed8efb2
# 搜索 def simple_decoder_fn_train
def dynamic_rnn_decoder(cell, decoder_fn, inputs=None, sequence_length=None,
                        number_attention_units=None,
                        parallel_iterations=None, swap_memory=False,
                        time_major=False, scope=None, name=None):
  with ops.name_scope(name, "dynamic_rnn_decoder",
                      [cell, decoder_fn, inputs, sequence_length,
                       parallel_iterations, swap_memory, time_major, scope]):
    if inputs is not None:
      # Convert to tensor
      inputs = ops.convert_to_tensor(inputs) # (128, 11, 256), (128, 1, 256)

      print('inputs1: ', inputs)
      # Test input dimensions
      if inputs.get_shape().ndims is not None and (
          inputs.get_shape().ndims < 2):
        raise ValueError("Inputs must have at least two dimensions")
      # Setup of RNN (dimensions, sizes, length, initial state, dtype)
      if not time_major:
        # [batch, seq, features] -> [seq, batch, features]
        inputs = array_ops.transpose(inputs, perm=[1, 0, 2]) # (11, 128, 256), (1, 128, 256)
      print('inputs2: ', inputs)

      dtype = inputs.dtype
      # Get data input information
      input_depth = int(inputs.get_shape()[2]) # 256
      #if number_attention_units is not None:
      #  input_depth += number_attention_units
      batch_depth = inputs.get_shape()[1].value # 128
      max_time = inputs.get_shape()[0].value # 11
      if max_time is None:
        max_time = array_ops.shape(inputs)[0]
      # Setup decoder inputs as TensorArray
      inputs_ta = tensor_array_ops.TensorArray(dtype, size=max_time)
      inputs_ta = inputs_ta.unstack(inputs) # 把inputs展开成max_time次循环，

    # https://zhuanlan.zhihu.com/p/39561455 loop_fn的作用
    def loop_fn(time, cell_output, cell_state, loop_state):
      if cell_state is None:  # first call, before while loop (in raw_rnn)
        if cell_output is not None:
          raise ValueError("Expected cell_output to be None when cell_state "
                           "is None, but saw: %s" % cell_output)
        if loop_state is not None:
          raise ValueError("Expected loop_state to be None when cell_state "
                           "is None, but saw: %s" % loop_state)
        context_state = None
      else:  # subsequent calls, inside while loop, after cell excution
        if isinstance(loop_state, tuple):
          (done, context_state) = loop_state
        else:
          done = loop_state
          context_state = None

      # call decoder function
      if inputs is not None:  # training; inputs是序列化没有encode的原始x
        # get next_cell_input
        print('cell_state11', cell_state) # 第一次cell_state是none，以后是(128, 256)
        if cell_state is None:
          read_input = inputs_ta.read(0)
          print("first read:", read_input) # (128, 256)
          (next_done, next_cell_state, next_cell_input, emit_output,
           next_context_state) = decoder_fn(time, cell_state, read_input, # decoder_fn两种都调用了，分别是1，1，2，2，
                                            cell_output, context_state)
          print("first next_cell_input:", next_cell_input) # (128, 256)
        else:
          if batch_depth is not None:
            batch_size = batch_depth
          else:
            batch_size = array_ops.shape(done)[0]
          read_input = control_flow_ops.cond(
              math_ops.equal(time, max_time),
              lambda: array_ops.zeros([batch_size, input_depth], dtype=dtype),
              lambda: inputs_ta.read(time))
          print("second read:", read_input) # (128, 256)
          (next_done, next_cell_state, next_cell_input, emit_output,
           next_context_state) = decoder_fn(time, cell_state, read_input, # decoder_fn两种都调用了，分别是1，1，2，2，
                                            cell_output, context_state, reuse=True)
          print("second next_cell_input:", next_cell_input)
      else:  # inference; 这个分支没有走，都是上个分支
        # next_cell_input is obtained through decoder_fn
        print('cell_state12', cell_state)
        (next_done, next_cell_state, next_cell_input, emit_output,
         next_context_state) = decoder_fn(time, cell_state, None, cell_output,
                                          context_state)

      # check if we are done
      if next_done is None:  # training
        next_done = time >= sequence_length

      # build next_loop_state
      if next_context_state is None:
        next_loop_state = next_done
      else:
        next_loop_state = (next_done, next_context_state)

      return (next_done, next_cell_input, next_cell_state,
              emit_output, next_loop_state)

    # Run raw_rnn function
    outputs_ta, state, _ = tf.nn.raw_rnn(
        cell, loop_fn, parallel_iterations=parallel_iterations,
        swap_memory=swap_memory, scope=scope)
    outputs = outputs_ta.stack()

    if not time_major:
      # [seq, batch, features] -> [batch, seq, features]
      outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
    return outputs, state

def simple_decoder_fn_train(encoder_state, context_fn=None, name=None):
  with ops.name_scope(name, "simple_decoder_fn_train", [encoder_state, context_fn]):
    pass

  def decoder_fn(time, cell_state, cell_input, cell_output, context_state, reuse=False):
    with ops.name_scope(name, "simple_decoder_fn_train",
                        [time, cell_state, cell_input, cell_output,
                         context_state]):
      print('decoder_fn1') # 调用两次，一次是first input，第二次是second input。后面才调用not train的decoder_fn
      if cell_state is None:  # first call, return encoder_state; 第一次调用使用encoder的hidden_final即这里encoder_state
        if context_fn is not None:
          cell_input, _ = context_fn(encoder_state, cell_input, reuse=reuse)
        return (None, encoder_state, cell_input, cell_output, context_state)
      else:
        if context_fn is not None:
          cell_input, _ = context_fn(cell_state, cell_input, reuse=reuse)
        return (None, cell_state, cell_input, cell_output, context_state)
  return decoder_fn


def decoder_rnn(cell, 
                inputs, # (128, 11, 256)
                enc_outputs, # (128, 11, 256)
                enc_final_states, # (128, 256)
                seq_length,  # 128
                hidden_dim, # 256
                num_glimpse, # 1
                batch_size,  # 128
                is_train,
                end_of_sequence_id=0, 
                initializer=None,
                max_length=None):
  with tf.variable_scope("decoder_rnn") as scope:
    print('decoder_rnn call') # 调用两次
    def attention(ref, query, with_softmax, scope="attention"):
      '''
      print('ref: ', ref) # (128, 11, 256)
      print('query: ', query) # (128, 256)
      '''

      with tf.variable_scope(scope):
        W_ref = tf.get_variable(
            "W_ref", [1, hidden_dim, hidden_dim], initializer=initializer) #  W_ref, [1, 256, 256]
        W_q = tf.get_variable(
            "W_q", [hidden_dim, hidden_dim], initializer=initializer) # W_q, [256, 256]
        v = tf.get_variable(
            "v", [hidden_dim], initializer=initializer) # v, [256]

        #ref = tf.Print(ref, [tf.shape(ref)], 'decoder_rnn ref: ') # [16 11 256]
        #ref = tf.Print(ref, [tf.shape(query)], 'decoder_rnn query: ') # [16 256]
        # (128, 11, 256) * [1, 输入256, 输出filter256]
        encoded_ref = tf.nn.conv1d(ref, W_ref, 1, "VALID", name="encoded_ref") # (128, 11, 256)
        # (128, 256) * [256, 256]
        encoded_query = tf.expand_dims(tf.matmul(query, W_q, name="encoded_query"), 1) # (128, 1, 256)
        tiled_encoded_Query = tf.tile(
            encoded_query, [1, tf.shape(encoded_ref)[1], 1], name="tiled_encoded_query") # (128, 1, 256) -> (128, 11, 256)
        # (128, 11, 256)
        scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query), [-1]) # (128, 11)
        if with_softmax:
          scores_s = tf.nn.softmax(scores) # (128, 11)
          return tf.nn.softmax(scores_s)
        else:
          return scores

    def glimpse(ref, query, scope="glimpse"):
      p = attention(ref, query, with_softmax=True, scope=scope) # (128, 11)
      alignments = tf.expand_dims(p, 2) # (128, 11, 1)
      # (128, 11, 1) * (128, 11, 256) -> (128, 11, 256)
      # 返回带权重的ref (128, 11, 256) -> (128, 256)
      return tf.reduce_sum(alignments * ref, [1])

    def output_fn(ref, query, num_glimpse):
      #ref = tf.Print(ref, [tf.shape(ref)], 'output_fn ref shape: ') # [16 11 256]
      #ref = tf.Print(ref, [query], 'output_fn query shape: ', summarize=20) # [16 256]
      if query is None:
        # 第二次调用有max_length值
        return tf.zeros([max_length], tf.float32) # only used for shape inference
      else:
        for idx in range(num_glimpse):
          print('output_fn ref: ', ref) # (128, 11, 256)
          print('output_fn query: ', query) # (128, 256) 两次一致
          query = glimpse(ref, query, "glimpse_{}".format(idx)) # 返回带权重的ref(128, 256)
          print('weight query: ', query) # (128, 256)
        return attention(ref, query, with_softmax=False, scope="attention") # (128, 11)

    def input_fn(sampled_idx):
      return tf.stop_gradient(
          tf.gather_nd(enc_outputs, index_matrix_to_pairs(sampled_idx)))

    if is_train:
      decoder_fn = simple_decoder_fn_train(enc_final_states)
    else:
      maximum_length = tf.convert_to_tensor(max_length, tf.int32)

      def decoder_fn(time, cell_state, cell_input, cell_output, context_state, reuse=True):
        # 对rnn的输出做处理给作为下一个timestep rnn的输入, 第一次调用为对rnn的输入做预处理
        # cell_input is none
        # 第一次调用cell_output和cell_state都为空
        print('decoder_fn2 cell_state: ', cell_state) # 初始None, (128, 256)
        print('decoder_fn2 cell_input: ', cell_input) # 初始(128, 256), (128, 256)
        print('decoder_fn2 cell_output: ', cell_output) # 初始None, (128, 256)
        cell_output = output_fn(enc_outputs, cell_output, num_glimpse)
        if cell_state is None:
          cell_state = enc_final_states
          next_input = cell_input
          done = tf.zeros([batch_size,], dtype=tf.bool)
        else:
          print('decoder_fn2 cell_output2: ', cell_output) # (128, 11)
          sampled_idx = tf.cast(tf.argmax(cell_output, 1), tf.int32)
          next_input = input_fn(sampled_idx)
          done = tf.equal(sampled_idx, end_of_sequence_id)

        done = tf.cond(tf.greater(time, maximum_length),
          lambda: tf.ones([batch_size,], dtype=tf.bool),
          lambda: done)
        return (done, cell_state, next_input, cell_output, context_state)

    # 调用两次
    outputs, (final_state, final_context_state) = \
        dynamic_rnn_decoder(cell, decoder_fn, inputs=inputs,
                            sequence_length=seq_length, scope=scope)
    #outputs = tf.Print(outputs, [tf.shape(outputs)], 'outputs shape decoder: ') # [128 11 256]
    #outputs = tf.Print(outputs, [tf.shape(final_state)], 'final_state shape decoder: ') # [128 256], [16 256]

    if is_train:
      transposed_outputs = tf.transpose(outputs, [1, 0, 2])
      #transposed_outputs = tf.Print(transposed_outputs, [tf.shape(enc_outputs)], 'decoder_rnn enc_outputs: ') # [16 11 256]
      #transposed_outputs = tf.Print(transposed_outputs, [tf.shape(transposed_outputs)], 'decoder_rnn transposed_outputs: ') # [11 16 256]
      fn = lambda x: output_fn(enc_outputs, x, num_glimpse) 
      # transposed_outputs的shape是[n, 128, 256], 每次执行fn，需要dim=0拆分，[128, 256]
      # map_fn会把transposed_outputs从第一维展开变成11个[16, 256], 实验验证了。
      #transposed_outputs = tf.Print(transposed_outputs, [transposed_outputs[0]], 'decoder_rnn transposed_outputs: ', summarize=20)
      outputs = tf.transpose(tf.map_fn(fn, transposed_outputs), [1, 0, 2]) # transposed_outputs -> lambda x
      #outputs = tf.Print(outputs, [tf.shape(outputs)], 'decoder_rnn outputs: ') # [16 11 11]


    return outputs, final_state, final_context_state

def trainable_initial_state(batch_size, state_size,
                            initializer=None, name="initial_state"):
  flat_state_size = nest.flatten(state_size)

  if not initializer:
    flat_initializer = tuple(tf.zeros_initializer for _ in flat_state_size) # 初始值都是0
  else:
    flat_initializer = tuple(tf.zeros_initializer for initializer in flat_state_size)

  names = ["{}_{}".format(name, i) for i in range(len(flat_state_size))]
  tiled_states = []

  for name, size, init in zip(names, flat_state_size, flat_initializer):
    shape_with_batch_dim = [1, size]
    initial_state_variable = tf.get_variable(
        name, shape=shape_with_batch_dim, initializer=init()) # 使用tf.zeros_initializer，初始值都是0.

    tiled_state = tf.tile(initial_state_variable,
                          [batch_size, 1], name=(name + "_tiled"))
    tiled_states.append(tiled_state)

  return nest.pack_sequence_as(structure=state_size,
                               flat_sequence=tiled_states)

def index_matrix_to_pairs(index_matrix):
  # [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]], 
  #                        [[0, 2], [1, 3], [2, 1]]]
  replicated_first_indices = tf.range(tf.shape(index_matrix)[0]) # (128,)
  rank = len(index_matrix.get_shape()) # (128, 10) 2
  if rank == 2:
    replicated_first_indices = tf.tile(
        tf.expand_dims(replicated_first_indices, dim=1), #[128] -> [128, 1] -> [128, 10] -> [[000000000000], [1111111111111], [2222222222222], ...[]]
        [1, tf.shape(index_matrix)[1]])
  return tf.stack([replicated_first_indices, index_matrix], axis=rank) # [[[0,5], [0,1]....[0,10]], [[1,6],[1,0], ...[1,10]], ...[[128,5], [128,0],...[128,10]]]
