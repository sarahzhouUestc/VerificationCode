#--------------------定长训练的参数--------------------

CODE_LEN = 6        #验证码文本长度
IMG_WIDTH = 300
IMG_HEIGHT = 70
CHAR_SET_LEN = 62
FONT_DIR = './fonts/'
IMG_DIR_TRAIN = './images/train/'
IMG_DIR_VAL = './images/val/'
IMG_DIR_TEST = './images/test/'
CHAR_SET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
FONT_LIST = ['arialuni.ttf', 'Maydana Ika Putri.ttf', 'calibri.ttf', 'Monaco.ttf', 'Arial_0.ttf']


BATCH_SIZE = 128
NUM_CLASSES = 62    #10个数字、26个小写字母、26个大写字母
DATA_SIZE = 20000
NUM_EPOCHS = 15000


INPUT_WIDTH = 256       #网络输入图片尺寸
INPUT_HEIGHT = 64

LEARNING_RATE_BASE=0.0001    #学习率
LEARNING_RATE_DECAY=0.9

REGULARIZATION_RATE=0.0001  #正则化参数

MOVING_AVERAGE_DECAY=0.99   #滑动平均衰减率

MODEL_SAVE_PATH="./model/"  #模型保存
MODEL_NAME="vericode.ckpt"

BN_EPS=1e-5
