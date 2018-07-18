"""
装配输入数据,这个文件针对固定长度验证码的处理
"""
import cv2
import numpy as np
import os, random

from utils.CONFIG import *


def _char2vec(c):
    vec = np.zeros((CHAR_SET_LEN))   #62长
    for i in range(CHAR_SET_LEN):
        if CHAR_SET[i] == c:
            vec[i] = 1
    return vec

def _one_hot_vec(text):
    vecs = np.zeros((CODE_LEN * CHAR_SET_LEN))
    for i in range(CODE_LEN):
        vec = _char2vec(list(text)[i])
        vecs[i * CHAR_SET_LEN : (i + 1) * CHAR_SET_LEN] = np.copy(vec)
    return vecs

def vec2text(vec):
    pred = np.reshape(vec, [CODE_LEN, CHAR_SET_LEN])
    text = []
    for i in range(pred.shape[0]):
        text.append(CHAR_SET[np.argmax(pred[i])])
    return ''.join(text)


def _text2vec(text):         #多目标label向量
    vec = np.zeros(CHAR_SET_LEN)
    for i in range(CODE_LEN):
        idx = CHAR_SET.index(text[i])
        vec[idx] = 1
    return vec


def get_train_batch():
    for _, _, files in os.walk(IMG_DIR_TRAIN): #fs是图片文件名列表
        batch_img = []
        batch_labels = []       #one-hot的形式
        while True:     #f是图片文件名
            rdn = random.randint(0, len(files)-1)
            f_path = os.path.join(IMG_DIR_TRAIN, files[rdn])
            img = cv2.imread(f_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #转为灰度图，因为文本与颜色没有关系
            img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
            batch_img.append(img)

            text = files[rdn].split('_')[0]      #分离出图片中的文本
            # batch_labels.append(_text2vec(list(text)))
            batch_labels.append(_one_hot_vec(text))

            if len(batch_img)==BATCH_SIZE and len(batch_labels)==BATCH_SIZE:
                yield np.array(batch_img), np.array(batch_labels)
                batch_img, batch_labels = [], []

#获取所有的验证数据
def get_val_data():
    batch_img, batch_labels, val_labels = [], [], []
    for _, _, files in os.walk(IMG_DIR_VAL):
        for f in files:
            f_path = os.path.join(IMG_DIR_VAL, f)
            img = cv2.imread(f_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # 转为灰度图，因为文本与颜色没有关系
            img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
            batch_img.append(img)
            text = f.split('_')[0]      # 分离出图片中的文本
            batch_labels.append(_one_hot_vec(text))
            # batch_labels.append(_text2vec(list(text)))
            val_labels.append(text)
    return np.array(batch_img), np.array(batch_labels), np.array(val_labels)

#获取所有的测试数据
def get_test_data():
    batch_img, batch_labels, text_labels = [], [], []
    for _, _, files in os.walk(IMG_DIR_TEST):
        for f in files:
            f_path = os.path.join(IMG_DIR_TEST, f)
            img = cv2.imread(f_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # 转为灰度图，因为文本与颜色没有关系
            img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
            batch_img.append(img)
            text = f.split('_')[0]      # 分离出图片中的文本
            batch_labels.append(_one_hot_vec(text))
            # batch_labels.append(_text2vec(list(text)))
            text_labels.append(text)
    return np.array(batch_img), np.array(batch_labels), np.array(text_labels)