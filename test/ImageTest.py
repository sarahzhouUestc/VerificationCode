# -*- coding: utf-8 -*-
from PIL import Image
import glob, os

img = Image.open('./test/face.png').convert('RGBA')
# print img.size
# print img.mode
# img.rotate(45).show()

'''
# alpha_composite
img1 = Image.open('./test/me.JPG').convert('RGBA')
img2 = Image.open('./test/flower2.JPG').convert('RGBA')
img2 = img2.resize(img1.size)
a_com = Image.alpha_composite(img1, img2)
a_com.save('./test/alpha.JPG')
print img1.mode, img2.mode
print img1.size, img2.size

new_img2 = img2.resize(img1.size)
new_img2.save('./test/new_bear.jpg')
print img1.size, new_img2.size
img3 = Image.alpha_composite(img1, new_img2)
img3.save('./test/alpha_composite.jpg')

# blend
blend_img = Image.blend(img1, img2, 0.8)
blend_img.save('./test/blend.jpg')


# composite
img1 = Image.open('./test/me.JPG').convert('RGBA')
img1 = img1.resize((960, 1280))
img2 = Image.open('./test/white.JPG').convert('RGBA')
img2 = img2.resize((960, 1280))
img3 = Image.open('./test/white2.JPG').convert('RGBA')
print img1.size,img2.size,img3.size
composite = Image.composite(img1,img2,img3)
composite.save('./test/composite.jpg')
img1 = Image.open('./test/me.JPG').convert('L')
img1.save('./test/L.JPG')
'''

# # new
# im = Image.new('RGB', (960, 1280), (100, 80, 125))
# # im.save('./test/new.jpg')
# rgb2xyz = (
#     0.412453, 0.357580, 0.180423, 0,
#     0.212671, 0.715160, 0.072169, 0,
#     0.019334, 0.119193, 0.950227, 0)
# out = im.convert('RGB', rgb2xyz)
# out.save('./test/rgb.jpg')
# print out.getcolors()
# print out.getdata()
# print out.getextrema()
# print out.getpixel((50, 70))
#
# out.putalpha(6)
# out.save('./test/rgb1.jpg')
# # out.show()
#
# im1 = out.split()
# print '==', out.tell()
# out.thumbnail((1000, 100))
# out.transpose(Image.TRANSPOSE).show()

img1 = Image.open('./test/a.png').convert('L')
img1.show()
img1.save('./test/a.bmp')

