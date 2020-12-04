#综合图像识别和文字分类
#导入所需要的库
import numpy as np
import os, sys
sys.path.append('textcnn')#自定义引用模块的路径
from textcnn.predict import RefuseClassification
from classify_image import *


class RafuseRecognize():
    
    def __init__(self):
        
        self.refuse_classification = RefuseClassification()
        self.init_classify_image_model()
        self.node_lookup = NodeLookup(uid_chinese_lookup_path='./data/imagenet_2012_challenge_label_chinese_map.pbtxt', 
                                model_dir = '/tmp/imagenet')#将中文对照表转换为人类可读ID
        
        
    def init_classify_image_model(self):
        
        create_graph('/tmp/imagenet')#从保存的GraphDef文件创建一个图形并返回一个保存程序。

        self.sess = tf.Session()#建立会话
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')
        
        
    def recognize_image(self, image_data):
        
        predictions = self.sess.run(self.softmax_tensor,#执行该变量
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)#从数组的形状中删除单维度条目，即把shape中为1的维度去掉

        top_k = predictions.argsort()[-5:][::-1]#从小到大排序
        result_list = []
        for node_id in top_k:
            human_string = self.node_lookup.id_to_string(node_id)#转换为string类型
            #print(human_string)
            human_string = ''.join(list(set(human_string.replace('，', ',').split(','))))#添加分隔符
            #print(human_string)
            classification = self.refuse_classification.predict(human_string)#对用户输入的进行分类
            result_list.append('%s  =>  %s' % (human_string, classification))#多种结果产生的列表
            
        return '\n'.join(result_list)
        

if __name__ == "__main__":
    if len(sys.argv) == 2:
        test = RafuseRecognize()
        image_data = tf.gfile.FastGFile(sys.argv[1], 'rb').read()
        res = test.recognize_image(image_data)
        print('classify:\n%s' %(res))
