#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-01-11 19:55:46
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-04-11 16:41:48

import pickle

import util


class DbItem:
    def __init__(self, label, feature, face_rgb):
        self.label = label
        self.feature = feature
        self.face_rgb = face_rgb


class DbManager:
    def __init__(self):
        self.__db = []

    def __del__(self):
        self.clear()

    def clear(self):
        self.__db = []

    def load(self, db_file):
        with open(db_file, 'r') as reader:
            self.__db = pickle.load(reader)

    def save(self, db_file):
        with open(db_file, 'w') as writer:
            pickle.dump(self.__db, writer)

    def search_db(self, feature, max_num, threshold, score_func=util.neg_euclidean_distance):
        score_list = []
        for item in self.__db:
            similarity = score_func(feature, item.feature)
            if similarity > threshold:
                score_list.append((item, similarity))
        if len(score_list) == 0:
            return 0, []
        sorted_score_list = sorted(
            score_list, key=lambda p: p[1], reverse=True)
        if max_num > len(sorted_score_list):
            max_num = len(sorted_score_list)
        return max_num, sorted_score_list[:max_num]

    def append_db(self, face_rgb, feature, label):
        self.__db.append(DbItem(label, feature, face_rgb))
