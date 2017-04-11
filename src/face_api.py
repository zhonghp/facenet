#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-04-11 15:48:14
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-04-11 17:53:56

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from scipy import misc

import db
import config
import facenet
import align.detect_face


class FaceAPI:
    def __init__(self):
        # configurations for detection.
        self.__gpu_memory_fraction = 0.25
        self.__minsize = 20
        self.__threshold = [0.6, 0.7, 0.7]
        self.__factor = 0.709
        self.__margin = 32
        self.__image_size = 160

        # configurations for extraction.
        self.__model_dir = config.model_dir
        self.__embedding_size = 128

        self.__db = db.DbManager()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.__gpu_memory_fraction)
        self.__detect_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with self.__detect_sess.as_default():
            self.__pnet, self.__rnet, self.__onet = align.detect_face.create_mtcnn(self.__detect_sess, None)

        self.__extract_sess = tf.Session()
        with self.__extract_sess.as_default():
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(self.__model_dir))
            facenet.load_model(self.__model_dir, meta_file, ckpt_file)
            self.__images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.__embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.__phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = self.__images_placeholder.get_shape()[1]
            assert image_size == self.__image_size

            embedding_size = self.__embeddings.get_shape()[1]
            assert embedding_size == self.__embedding_size

    def align_face(self, img_rgb, bbox):
        assert bbox is not None
        img_size = np.asarray(img_rgb.shape)[0:2]

        det = np.squeeze(bbox)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - self.__margin/2, 0)
        bb[1] = np.maximum(det[1] - self.__margin/2, 0)
        bb[2] = np.minimum(det[2] + self.__margin/2, img_size[1])
        bb[3] = np.minimum(det[3] + self.__margin/2, img_size[0])
        cropped = img_rgb[bb[1]:bb[3], bb[0]:bb[2], :]
        scaled = misc.imresize(cropped, (self.__image_size, self.__image_size), interp='bilinear')
        return scaled

    def detect_all_faces(self, img_rgb):
        bboxes, _ = align.detect_face.detect_face(img_rgb, self.__minsize, self.__pnet, self.__rnet, self.__onet, self.__threshold, self.__factor)
        nrof_faces = bboxes.shape[0]
        if nrof_faces > 0:
            det = bboxes[:, 0:4]
            return det
        return None

    def detect_largest_face(self, img_rgb):
        bboxes, _ = align.detect_face.detect_face(img_rgb, self.__minsize, self.__pnet, self.__rnet, self.__onet, self.__threshold, self.__factor)
        nrof_faces = bboxes.shape[0]
        if nrof_faces > 0:
            det = bboxes[:, 0:4]
            img_size = np.asarray(img_rgb.shape)[0:2]
            if nrof_faces > 1:
                bbox_size = (det[:, 2]-det[:, 0]) * (det[:, 3]-det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack([(det[:, 0]+det[:, 2])/2 - img_center[1], (det[:, 1]+det[:, 3])/2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bbox_size - offset_dist_squared*2.0)
                det = det[index, :]
            return det
        return None

    def __get_face_feature(self, face_rgb):
        images = np.zeors((1, self.__image_size, self.__image_size, 3))
        images[0, :,:,:] = facenet.prewhiten(face_rgb)
        feed_dict = {self.__images_placeholder: images, self.__phase_train_placeholder: False}

        embed_array = np.zeros((1, self.__embedding_size))
        embed_array[:, :] = self.__extract_sess.run(self.__embeddings, feed_dict=feed_dict)
        return embed_array[0, :]

    def __get_face_feature(self, img_rgb, bbox):
        face_rgb = self.align_face(img_rgb, bbox)
        feature = self.__get_face_feature(face_rgb)
        return face_rgb, feature

    def save_db(self, filename="train.pkl"):
        self.__db.save(filename)

    def load_db(self, filename="train.pkl"):
        self.__db.load(filename)

    def clear_db(self):
        self.__db.clear()

    def __search_db_by_face(self, img_rgb, bbox, max_num, threshold):
        _, feature = self.__get_face_feature(img_rgb, bbox)
        return self.__db.search_db(feature, max_num, threshold)

    def search_db(self, img_rgb, max_num=3, threshold=-1.25):
        bboxes = self.detect_all_faces(img_rgb)
        if bboxes is None:
            return
        for bbox in bboxes:
            count, score_list = self.__search_db_by_face(img_rgb, bbox, max_num, threshold)
            yield bbox, count, score_list

    '''
    def __append_face_to_db(self, img_rgb, bbox, label):
        if isinstance(label, str):
            label = label.decode('utf-8')
        face_rgb, feature = self.__get_face_feature(img_rgb, bbox)
        self.__db.append_db(face_rgb, feature, label)

    def append_db(self, img_rgb, label):
        bboxes = self.detect_all_faces(img_rgb)
        if bboxes is None:
            return
        for bbox in bboxes:
            self.__append_face_to_db(img_rgb, bbox, label)
    '''

    def append_face_to_db(self, face_rgb, label):
        if isinstance(label, str):
            label = label.decode('utf-8')
        feature = self.__get_face_feature(face_rgb)
        self.__db.append_db(face_rgb, feature, label)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print('Usage1: python face_api.py [in_folder] [out_folder] [1]')
        print('Usage2: python face_api.py [db_folder] [test_folder] [2]')
        sys.exit(-1)

    import os
    mode = int(sys.argv[3])
    if mode == 1:
        in_folder = sys.argv[1].strip()
        out_folder = sys.argv[2].strip()
        face_api = FaceAPI()
        for subdir in os.listdir(in_folder):
            folder = os.path.join(in_folder, subdir)
            for path in os.listdir(folder):
                path = os.path.join(folder, path)
                img = misc.imread(path)
                assert img.ndim >= 2
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:, :, 0:3]
                bboxes = face_api.detect_all_faces(img)
                if bboxes is None:
                    print('No detected face in:', path)
                    continue

                print(len(bboxes), 'faces detected in', path)
                for idx, bbox in enumerate(bboxes):
                    face = face_api.align_face(img, bbox)
                    out_file = os.path.join(out_folder, subdir)
                    if not os.path.exists(out_file):
                        os.makedirs(out_file)
                    out_file = os.path.join(out_file, str(idx)+'.jpg')
                    misc.imsave(out_file, face)
    elif mode == 2:
        import time
        db_folder = sys.argv[1].strip()
        test_folder = sys.argv[2].strip()
        face_api = FaceAPI()
        for subdir in os.listdir(db_folder):
            folder = os.path.join(in_folder, subdir)
            for path in os.listdir(folder):
                path = os.path.join(folder, path)
                img = misc.imread(path)
                assert img.ndim >= 2
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:, :, 0:3]
                face_api.append_face_to_db(img, subdir)
        face_api.save_db()

        writer = open('scores.txt', 'w')

        face_api = FaceAPI()
        face_api.load_db()
        start = time.time()
        for subdir, dirs, files in os.walk(test_folder):
            for path in files:
                (label, fname) = (os.path.basename(subdir), path)
                (name, ext) = os.path.splitext(fname)

                img = misc.imread(os.path.join(subdir, fname))
                assert img.ndim >= 2
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:, :, 0:3]

                print("test label: {}".format(label))
                print("test filename: {}".format(fname))
                max_label = ''
                max_similarity = -1000
                for idx, result in enumerate(face_api.search_db(img)):
                    bbox, count, score_list = result
                    print("\tsearch {} faces in db".format(count))
                    print("\tsimilarity: {}".format(score_list))
                    for (item, similarity) in score_list:
                        label = item.label.encode('utf-8')
                        if similarity >= max_similarity:
                            max_label = label
                            max_similarity = similarity
                writer.writer(name + '\t' + max_label + '\t' + str(max_similarity) + '\n')
        end = time.time()
        print(end-start, 's')
        writer.flush()
        writer.close()

    else:
        pass
