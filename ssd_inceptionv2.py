import os
import sys
import label_map_util
import cv2
import time
import commands

o = commands.getoutput('/home/test/ssd/gpu.sh')
available_gpu = o.split('output_gpu::')[1].split(':')[2]

os.environ['CUDA_VISIBLE_DEVICES']=available_gpu

import numpy as np
import tensorflow as tf


class SSD_XIILAB:
    
    img_list = []
    result = []

    def __init__(self):
        
        if tf.__version__ != '1.4.0':
            raise ImportError('Upgrade your tensorflow version to 1.4.0')

        graph_file = '/home/test/ssd/ssd_inception_v2_mscoco_graph.pb'
        labels_file = '/home/test/ssd/mscoco_label_map.pbtxt'
        num_classes = 90
        
        detection_graph = self.load_model(graph_file)
        print ('Load Model from file : {}'.format(graph_file))
        
        #self.sess = tf.Session(graph=detection_graph)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        self.sess = tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options))
        
        print ('Session created')
        
        self.label_map = label_map_util.load_labelmap(labels_file)
        categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        print ('Load Labels from file : {}'.format(labels_file))

        
        
    def load_model(self, graph_file):
        
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
            self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        return detection_graph        
    
    
    def append_to_list(self, nparr):

	nparr = nparr.ravel().reshape(720,1280,3)
        self.img_list.append(nparr)
        
    def inference(self):
        
        s_time = time.time()
        
        input_images = np.zeros((len(self.img_list), 720, 1280, 3), dtype=np.int)
        for i in range(len(self.img_list)):
            input_images[i] = cv2.resize(self.img_list[i], (1280,720))
            
        (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: input_images})
       
        print ("Elapsed : {}".format(time.time()-s_time))
       
        return self.generate_result(boxes, scores, classes)
        
    def generate_result(self, boxes, scores, classes):
        # each frame : len(boxes) 
        total_results = []

        for frame_num in range(len(scores)):
            ## each frame

            total_results.append([])

            each_frame_scores = []

            for detect_num in range(len(scores[frame_num])):
                ## detection count 
                if scores[frame_num][detect_num] > 0.5:
                    each_frame_scores.append(scores[frame_num][detect_num])
                else:
                    break

            for detection in range(len(each_frame_scores)):

                total_results[frame_num].append({})
                total_results[frame_num][detection]['type'] = self.category_index[classes[frame_num][detection]]['name']
                ymin, xmin, ymax, xmax = boxes[frame_num][detection]
                left, right, top, bottom = (xmin * 1280, xmax*1280, ymin*720, ymax*720)
                
                x = int(left)
                y = int(top)
                w = int(right - left)
                h = int(bottom - top)

		#cropped_img = self.img_list[frame_num][y:y+h, x:x+w]
		#r, buf = cv2.imencode('.jpg', cropped_img)
                
                
                total_results[frame_num][detection]['x'] = x
                total_results[frame_num][detection]['y'] = y
                total_results[frame_num][detection]['w'] = w
                total_results[frame_num][detection]['h'] = h
                #total_results[frame_num][detection]['img'] = buf
                '''
                total_results[frame_num][detection]['left'] = left
                total_results[frame_num][detection]['right'] = right
                total_results[frame_num][detection]['top'] = top
                total_results[frame_num][detection]['bottom'] = bottom
                '''
                
        self.img_list = []

        return total_results
              
                
                
                
                
                
            

