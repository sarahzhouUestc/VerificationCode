"""
装配不定长验证码的数据
"""

from utils.CONFIG_M import *
import numpy as np
import os,random,cv2


# 构造 SparseTensor 供ctc_loss使用
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend([CHAR_SET.index(i) for i in list(seq)])

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


#获取不定长验证码的训练数据
def get_train_batch():
    for _, _, files in os.walk(IMG_DIR_TRAIN): #fs是图片文件名列表
        batch_img = []
        batch_labels = []
        while True:             #f是图片文件名
            rdn = random.randint(0, len(files)-1)
            f_path = os.path.join(IMG_DIR_TRAIN, files[rdn])
            img = cv2.imread(f_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #转为灰度图，因为文本与颜色没有关系
            img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))

            batch_img.append(np.transpose(img))
            batch_labels.append(files[rdn].split('_')[0])   #文本label

            if len(batch_img)==BATCH_SIZE and len(batch_labels)==BATCH_SIZE:
                sparse_targets = sparse_tuple_from(batch_labels)
                seq_len = np.ones(len(batch_img)) * INPUT_WIDTH
                yield np.array(batch_img), sparse_targets, seq_len
                batch_img, batch_labels = [], []


def get_val_data():
    batch_img = []
    batch_labels = []
    for _, _, files in os.walk(IMG_DIR_VAL):
        for f in files:
            f_path = os.path.join(IMG_DIR_VAL, f)
            img = cv2.imread(f_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # 转为灰度图，因为文本与颜色没有关系
            img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
            batch_img.append(np.transpose(img))
            batch_labels.append(f.split('_')[0])  # 文本label
            sparse_targets = sparse_tuple_from(batch_labels)
            seq_len = np.ones(len(batch_img)) * INPUT_WIDTH
    return np.array(batch_img), sparse_targets, seq_len