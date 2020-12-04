# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.
Run image classification with Inception trained on ImageNet 2012 Challenge data
set.
This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.
Change the --image_file argument to any jpg image to compute a
classification of that image.
Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.
https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import#加入绝对引入这个新特性
from __future__ import division#导入精确除法
from __future__ import print_function#即使在python2.X，使用print就得像python3.X那样加括号使用。

#导入所需要的模块
import argparse#命令行解析的标准模块
import os.path#获取文件的属性
import re#通过正则表达式是用来匹配处理字符串的
import sys#提供对解释器使用或维护的一些变量的访问
import tarfile#对文件进行压缩打包等操作

import numpy as np
from six.moves import urllib#Urllib是python内置的HTTP请求库
# import tensorflow as tf
import tensorflow.compat.v1 as tf
FLAGS = None #初始化全局常量

# pylint: disable=line-too-long
#模型下载地址
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

#用别人训练好的模型进行图像分类
#将类别id转换为人类易读的标签
class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self, 
                uid_chinese_lookup_path, 
                model_dir, 
                label_lookup_path=None,
                uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          model_dir, 'imagenet_synset_to_human_label_map.txt')
    #self.node_lookup = self.load(label_lookup_path, uid_lookup_path)
    self.node_lookup = self.load_chinese_map(uid_chinese_lookup_path)
    

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.
    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.
    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
	#加载分类字符串，对应分类文件的名字
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    #p = re.compile(r'[n\d]*[ \S,]*')
    p = re.compile(r'(n\d*)\t(.*)')
    for line in proto_as_ascii_lines:#一行行读取数据
      parsed_items = p.findall(line)
      print(parsed_items)
	  #获取分类编号和分类名称
      uid = parsed_items[0]
      human_string = parsed_items[1]
	  #保存编号字符串--------与分类名称映射关系
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
	#加载分类字符串-------对应分类编号的文件
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
	    #获取分类编号
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
	    #获取编号字符串
        target_class_string = line.split(': ')[1]
		#保存编号字符串--------与分类名称映射关系
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name
    
  def load_chinese_map(self, uid_chinese_lookup_path):
    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_chinese_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'(\d*)\t(.*)')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      #print(parsed_items)
      uid = parsed_items[0][0]
      human_string = parsed_items[0][1]
      uid_to_human[int(uid)] = human_string
    
    return uid_to_human
  #传入分类编号，返回分类名称
  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


#读取训练好的Inception-v3模型来创建graph
def create_graph(model_dir):
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  #创建一个图来存放训练好的模型
  with tf.gfile.FastGFile(os.path.join(
      model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


#读取图片，创建graph
def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph(FLAGS.model_dir)

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
	
	#Inception-v3模型的最后一层softmax的输出
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
	#输入图像数据，得到softmax概率值
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> chinese string lookup.
    node_lookup = NodeLookup(uid_chinese_lookup_path='./data/imagenet_2012_challenge_label_chinese_map.pbtxt', \
                                model_dir=FLAGS.model_dir)
	#取出概率最大的值
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
	  #获取分类名称
      human_string = node_lookup.id_to_string(node_id)
	  #获取该分类的置信度
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
      #print('node_id: %s' %(node_id))


#下载并提取模型的tar文件
#如果我们使用的pretrained模型已经不存在，这个函数会从tensorflow.org网站下载它并解压缩到一个目录。
def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

#调用函数
def main(_):
  maybe_download_and_extract()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  run_inference_on_image(image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()  #创建解析器
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(     #添加参数
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
