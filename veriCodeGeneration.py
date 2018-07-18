# -*- coding: utf-8 -*-
import os
import random
import uuid

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from utils.CONFIG_M import *

def _gene_color1():
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def _gene_color2():
    return (random.randint(30, 125), random.randint(30, 125), random.randint(30, 125))


def _gene_img(dir):
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), (255, 255, 255))  #生成白色验证码图片
    font = ImageFont.truetype(os.path.join(FONT_DIR, FONT_LIST[random.randint(0, len(FONT_LIST)-1)]), 40) #加载字体文件，创建字体对象
    draw = ImageDraw.Draw(img)
    for x in range(IMG_WIDTH):
        for y in range(IMG_HEIGHT):
            draw.point((x, y), fill=_gene_color2())      #填充图片背景

    txt = random.sample(CHAR_SET, random.randint(CODE_LEN_MIN,CODE_LEN_MAX))

    start = (IMG_WIDTH-30*len(txt))/2  #计算text开始渲染的起始位置
    for i in range(0, len(txt)):
        draw.text((start + 30 * i, 10), txt[i], font=font, fill=_gene_color1())  #第一个参数是文本渲染的位置
    # draw.text((40, 10), txt, font=font, fill=gene_color1())

    for i in range(0, 20):      #生成直线干扰
        draw.line((random.randint(0, IMG_WIDTH + 1), random.randint(0, IMG_HEIGHT + 1)) + (random.randint(0, IMG_WIDTH + 1), random.randint(0, IMG_HEIGHT + 1)), fill=60)
    del draw
    if random.randint(0,1):     #随机模糊图片
        img = img.filter(ImageFilter.BLUR)
    # img.show()
    img.save(dir + ''.join(txt) + "_" + str(uuid.uuid4()) + ".jpg")
    return


if __name__ == '__main__':
    # 生成指定数量的验证码图片，训练集
    for i in range(0,200):
        _gene_img(IMG_DIR_VAL)

    # 验证集
    # for i in range(0,500):
    #     _gene_img(IMG_DIR_VAL)

    # # 测试集
    # for i in range(0,200):
    #     _gene_img(IMG_DIR_TEST)


