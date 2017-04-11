#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-04-11 10:36:30
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-04-11 11:29:04

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import argparse
import numpy as np
import tensorflow as tf

import facenet
import align.detect_face


def main(args):
    with tf.Graph().as_default():
        in_file1 = os.path.expanduser(args.in_file1)
        in_file2 = os.path.expanduser(args.in_file2)

        # align faces.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

        minsize = 20
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709

        img1 = misc.imread(in_file1)
        assert img1.ndim >= 2
        if img1.ndim == 2:
            img1 = facenet.to_rgb(img1)
        img1 = img1[:, :, 0:3]
        bboxes1, _ = align.detect_face.detect_face(img1, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces1 = bboxes1.shape[0]
        if nrof_faces1 > 0:
            det = bboxes1.shape[:, 0:4]
            img_size = np.asarray(img1.shape)[0:2]
            if nrof_faces1 > 1:
                bbox_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack([(det[:, 0]+det[:, 2])/2 - img_center[1], (det[:, 1]+det[:, 3])/2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bbox_size - offset_dist_squared*2.0)
                det = det[index, :]
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - args.margin/2, 0)
            bb[1] = np.maximum(det[1] - args.margin/2, 0)
            bb[2] = np.minimum(det[2] + args.margin/2, img_size[1])
            bb[3] = np.minimum(det[3] + args.margin/2, img_size[0])
            cropped = img1[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled1 = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
            misc.imsave('aligned1.jpg', scaled1)

        img2 = misc.imread(in_file2)
        assert img2.ndim >= 2
        if img2.ndim == 2:
            img2 = facenet.to_rgb(img2)
        img2 = img2[:, :, 0:3]
        bboxes2, _ = align.detect_face.detect_face(img2, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces2 = bboxes2.shape[0]
        if nrof_faces2 > 0:
            det = bboxes2.shape[:, 0:4]
            img_size = np.asarray(img2.shape)[0:2]
            if nrof_faces2 > 1:
                bbox_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack([(det[:, 0]+det[:, 2])/2 - img_center[1], (det[:, 1]+det[:, 3])/2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bbox_size - offset_dist_squared*2.0)
                det = det[index, :]
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - args.margin/2, 0)
            bb[1] = np.maximum(det[1] - args.margin/2, 0)
            bb[2] = np.minimum(det[2] + args.margin/2, img_size[1])
            bb[3] = np.minimum(det[3] + args.margin/2, img_size[0])
            cropped = img2[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled2 = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
            misc.imsave('aligned2.jpg', scaled2)

        images = np.zeros((2, args.image_size, image_size, 3))
        images[0, :,:,:] = facenet.prewhiten(scaled1)
        images[1, :,:,:] = facenet.prewhiten(scaled2)

        with tf.Session() as sess:
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            facenet.load_model(args.model_dir, meta_file, ckpt_file)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = images_placeholder.get_shape()[1]
            assert image_size == args.image_size

            embedding_size = embeddings.get_shape()[1]
            batch_size = args.batch_size
            emb_array = np.zeros((2, embedding_size))

            # images = facenet.load_data(paths, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[:, :] = sess.run(embeddings, feed_dict=feed_dict)

            print emb_array[0, :].shape, emb_array[1, :].shape
            dist = np.sum(np.square(np.subtract(emb_array[0, :], emb_array[1, :])), 1)
            print dist


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('in_file1', type=str, help='')
    parser.add_argument('in_file2', type=str, help='')
    parser.add_argument('model_dir', type=str, help='')
    parser.add_argument('--batch_size', type=int, help='', default=2)
    parser.add_argument('--file_ext', type=str, help='', default='png', choices=['jpg', 'png'])
    parser.add_argument('--image_size', type=int, help='', default=160)
    parser.add_argument('--margin', type=int, help='', default=32)
    parser.add_argument('--gpu_memory_fraction', type=float, help='', default=0.25)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))