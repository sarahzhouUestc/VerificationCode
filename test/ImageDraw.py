# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
import sysconfig as sys

im = Image.open(r"C:\temp\image\flower1.JPG")
draw = ImageDraw.Draw(im)
print(im.size)
print (0, 0) + im.size
print (0, im.size[1], im.size[0], 0)
draw.line((0, 0) + im.size, fill=128)
# draw.line((0, im.size[1], im.size[0], 0), fill=128)
del draw

# write to stdout
# im.show()

print("==================================")
