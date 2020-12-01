# Most of the codes are from 
# https://github.com/vshallc/PtrNets/blob/master/pointer/misc/tsp.py
import os
import re
import zipfile
import itertools
import threading
import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm
from collections import namedtuple

# import tensorflow as tf
from download import download_file_from_google_drive

GOOGLE_DRIVE_IDS = {
    'tsp5_train.zip': '0B2fg8yPGn2TCSW1pNTJMXzFPYTg',
    'tsp10_train.zip': '0B2fg8yPGn2TCbHowM0hfOTJCNkU',
    'tsp5-20_train.zip': '0B2fg8yPGn2TCTWNxX21jTDBGeXc',
    'tsp50_train.zip': '0B2fg8yPGn2TCaVQxSl9ab29QajA',
    'tsp20_test.txt': '0B2fg8yPGn2TCdF9TUU5DZVNCNjQ',
    'tsp40_test.txt': '0B2fg8yPGn2TCcjFrYk85SGFVNlU',
    'tsp50_test.txt.zip': '0B2fg8yPGn2TCUVlCQmQtelpZTTQ',
}

TSP = namedtuple('TSP', ['x', 'y', 'name'])

def length(x, y):
  return np.linalg.norm(np.asarray(x) - np.asarray(y))

# https://gist.github.com/mlalevic/6222750
def solve_tsp_dynamic(points):
  #calc all lengths
  all_distances = [[length(x,y) for y in points] for x in points]
  #initial value - just distance from 0 to every other point + keep the track of edges
  A = {(frozenset([0, idx+1]), idx+1): (dist, [0,idx+1]) for idx,dist in enumerate(all_distances[0][1:])}
  cnt = len(points)
  for m in range(2, cnt):
    B = {}
    for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
      for j in S - {0}:
        B[(S, j)] = min( [(A[(S-{j},k)][0] + all_distances[k][j], A[(S-{j},k)][1] + [j]) for k in S if k != 0 and k!=j])  #this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
    A = B
  res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
  return np.asarray(res[1]) + 1 # 0 for padding

def generate_one_example(n_nodes, rng):
  nodes = rng.rand(n_nodes, 2).astype(np.float32)
  solutions = solve_tsp_dynamic(nodes)
  return nodes, solutions

def read_paper_dataset(paths, max_length):
  x, y = [], []
  for path in paths:
    # tf.logging.info("Read dataset {} which is used in the paper..".format(path))
    length = max(re.findall('\d+', path))
    with open(path) as f:
      for l in tqdm(f):
        inputs, outputs = l.split(' output ') # 0.995783476828 0.483911605554 0.066505595509 0.907456718211 0.933906322372 0.386423468701 0.127099588148 0.922532693676 0.678496811784 0.473945600162 0.157880382709 0.528278317405 0.888452884988 0.19244502255 0.151154398879 0.184920056672 0.951855080104 0.351818978909 0.236746105286 0.385156197263 output 1 3 9 7 4 2 6 10 8 5 1 
        x.append(np.array(inputs.split(), dtype=np.float32).reshape([-1, 2]))
        y.append(np.array(outputs.split(), dtype=np.int32)[:-1]) # skip the last one
  return x, y

class TSPDataLoader(object):
  def __init__(self, config, rng=None):
    self.config = config
    self.rng = rng

    self.task = config.task.lower() # tsp
    self.batch_size = config.batch_size # 128
    self.min_length = config.min_data_length # 5
    self.max_length = config.max_data_length #10

    self.is_train = config.is_train # True
    self.use_terminal_symbol = config.use_terminal_symbol # True
    self.random_seed = config.random_seed # 123

    self.data_num = {}
    self.data_num['train'] = config.train_num # 1000000
    self.data_num['test'] = config.test_num # 1000

    self.data_dir = config.data_dir # data
    self.task_name = "{}_({},{})".format(
        self.task, self.min_length, self.max_length) # tsp_(5,10)

    self.data = None
    self.coord = None
    self.threads = None
    self.input_ops, self.target_ops = None, None
    self.queue_ops, self.enqueue_ops = None, None
    self.x, self.y, self.seq_length, self.mask = None, None, None, None

    paths = self.download_google_drive_file()
    if len(paths) != 0:
      self._maybe_generate_and_save(except_list=paths.keys())
      for name, path in paths.items():
        self.read_zip_and_update_data(path, name)
    else:
      self._maybe_generate_and_save()
    self._create_input_queue()

  def _create_input_queue(self, queue_capacity_factor=16):
    self.input_ops, self.target_ops = {}, {}
    self.queue_ops, self.enqueue_ops = {}, {}
    self.x, self.y, self.seq_length, self.mask = {}, {}, {}, {}

    for name in self.data_num.keys():
      print('name: ', name)
      self.input_ops[name] = tf.placeholder(tf.float32, shape=[None, None])
      self.target_ops[name] = tf.placeholder(tf.int32, shape=[None])

      min_after_dequeue = 1000
      capacity = min_after_dequeue + 3 * self.batch_size

      self.queue_ops[name] = tf.RandomShuffleQueue(
          capacity=capacity,
          min_after_dequeue=min_after_dequeue,
          dtypes=[tf.float32, tf.int32],
          shapes=[[self.max_length, 2,], [self.max_length]],
          seed=self.random_seed,
          name="random_queue_{}".format(name))
      self.enqueue_ops[name] = \
          self.queue_ops[name].enqueue([self.input_ops[name], self.target_ops[name]])

      inputs, labels = self.queue_ops[name].dequeue()

      seq_length = tf.shape(inputs)[0]
      #seq_length = tf.Print(seq_length, [seq_length], 'seq_length: ') # [10]
      if self.use_terminal_symbol:
        mask = tf.ones([seq_length + 1], dtype=tf.float32) # terminal symbol
        print('seq_lengthloader: ', seq_length) # () 标量不显示，目测是10
        print('maskloader: ', mask) # (11,)
      else:
        mask = tf.ones([seq_length], dtype=tf.float32)


      self.x[name], self.y[name], self.seq_length[name], self.mask[name] = \
          tf.train.batch(
              [inputs, labels, seq_length, mask],
              batch_size=self.batch_size,
              capacity=capacity,
              dynamic_pad=True,
              name="batch_and_pad")
      print('self.x[name]: ', self.x[name])
      print('self.seq_length[name]: ', self.seq_length[name])
      print('self.mask[name]: ', self.mask[name])

  def run_input_queue(self, sess):
    self.threads = []
    self.coord = tf.train.Coordinator()

    for name in self.data_num.keys():
      
      def load_and_enqueue(sess, name, input_ops, target_ops, enqueue_ops, coord):
        idx = 0
        while not coord.should_stop():
          feed_dict = {
              input_ops[name]: self.data[name].x[idx],
              target_ops[name]: self.data[name].y[idx],
          }
          sess.run(self.enqueue_ops[name], feed_dict=feed_dict)
          idx = idx+1 if idx+1 <= len(self.data[name].x) - 1 else 0

      args = (sess, name, self.input_ops, self.target_ops, self.enqueue_ops, self.coord)
      t = threading.Thread(target=load_and_enqueue, args=args)
      t.start()
      self.threads.append(t)
      # tf.logging.info("Thread for [{}] start".format(name))

  def stop_input_queue(self):
    self.coord.request_stop()
    self.coord.join(self.threads)
    # tf.logging.info("All threads stopped")

  def _maybe_generate_and_save(self, except_list=[]):
    self.data = {}

    for name, num in self.data_num.items():
      if name in except_list:
        # tf.logging.info("Skip creating {} because of given except_list {}".format(name, except_list))
        continue
      path = self.get_path(name)

      if not os.path.exists(path):
        # tf.logging.info("Creating {} for [{}]".format(path, self.task))

        x = np.zeros([num, self.max_length, 2], dtype=np.float32)
        y = np.zeros([num, self.max_length], dtype=np.int32)

        for idx in trange(num, desc="Create {} data".format(name)):
          n_nodes = self.rng.randint(self.min_length, self.max_length+ 1)
          nodes, res = generate_one_example(n_nodes, self.rng)
          x[idx,:len(nodes)] = nodes
          y[idx,:len(res)] = res

        np.savez(path, x=x, y=y)
        self.data[name] = TSP(x=x, y=y, name=name)
      else:
        # tf.logging.info("Skip creating {} for [{}]".format(path, self.task))
        tmp = np.load(path)
        self.data[name] = TSP(x=tmp['x'], y=tmp['y'], name=name)

  def get_path(self, name):
    return os.path.join(
        self.data_dir, "{}_{}={}.npz".format(
            self.task_name, name, self.data_num[name]))

  def download_google_drive_file(self):
    paths = {}
    for mode in ['train', 'test']:
      candidates = []
      candidates.append(
          '{}{}_{}'.format(self.task, self.max_length, mode))
      candidates.append(
          '{}{}-{}_{}'.format(self.task, self.min_length, self.max_length, mode))

      for key in candidates:
        for search_key in GOOGLE_DRIVE_IDS.keys():
          if search_key.startswith(key):
            path = os.path.join(self.data_dir, search_key)
            # tf.logging.info("Download dataset of the paper to {}".format(path))

            if not os.path.exists(path):
              download_file_from_google_drive(GOOGLE_DRIVE_IDS[search_key], path)
              if path.endswith('zip'):
                with zipfile.ZipFile(path, 'r') as z:
                  z.extractall(self.data_dir)
            paths[mode] = path

    # tf.logging.info("Can't found dataset from the paper!")
    return paths

  def read_zip_and_update_data(self, path, name):
    if path.endswith('zip'):
      filenames = zipfile.ZipFile(path).namelist()
      paths = [os.path.join(self.data_dir, filename) for filename in filenames]
    else:
      paths = [path]

    # x_list, [[0.995783476828, 0.483911605554, 0.066505595509, 0.907456718211, 0.933906322372, 0.386423468701], []] 每个数据20个值，总共训练数据的个数
    # y_list, [[1, 2, 1, ...], [...]]v bvv v     lll,
    x_list, y_list = read_paper_dataset(paths, self.max_length)
    print('x_list len: ', len(x_list)) # 1010000 
    print('x_list[0]: ', x_list[0]) 
    '''
    [[0.6786686  0.09137148]
     [0.8814321  0.8569943 ]
     [0.03080868 0.07900781]
     [0.8105855  0.12803143]
     [0.9510739  0.8646896 ]
     [0.7519406  0.33385053]
     [0.7111053  0.6228321 ]
     [0.85490894 0.37674022]
     [0.2556446  0.82974094]
     [0.0242689  0.00339981]]
     '''
    print('y_list[0]: ', y_list[0]) # [ 1  3 10  9  5  2  7  8  6  4]

    # [训练数据个数, 10, 2]
    # [训练数据个数, 10]
    x = np.zeros([len(x_list), self.max_length, 2], dtype=np.float32)
    y = np.zeros([len(y_list), self.max_length], dtype=np.int32)

    # 第i个数据, ([20个点坐标cat20个输出index])
    for idx, (nodes, res) in enumerate(tqdm(zip(x_list, y_list))):
      x[idx,:len(nodes)] = nodes # x的每一行是一组训练数据[num_trains, 20]
      y[idx,:len(res)] = res # [num_trains, 10]


    if self.data is None:
      self.data = {}

    # tf.logging.info("Update [{}] data with {} used in the paper".format(name, path))
    self.data[name] = TSP(x=x, y=y, name=name)
    print('self.data size: ', len(self.data[name])) # 3
    print('self.data[0] size: ', len(self.data[name][0])) # 1010000
    print('self.data[name][0][0]: ', self.data[name][0][0]) # [10, 2]坐标矩阵
    print('self.data[name][1][0]: ', self.data[name][1][0]) # [10] 输出
    print('self.data[name][2][0]: ', self.data[name][2]) # train


