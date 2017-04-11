#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-04-11 16:38:43
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-04-11 16:41:31

import numpy as np


def euclidean_distance(feat1, feat2):
    return np.sum(np.square(np.subtract(feat1, feat2)), 0)


def neg_euclidean_distance(feat1, feat2):
    return -1.0 * euclidean_distance(feat1, feat2)