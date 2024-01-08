from PIL import Image,ImageOps,ImageStat,ImageDraw
import os,os.path
import sys
import math
import numpy as np
import random

t_color = ['blue', 'red', 'yellow', 'green', 'violet']
t_color_codes = [[0,0,255],[255,0,0],[255,255,0],[0,128,0],[238,130,138]]
'''
aliceblue                      : #f0f8ff
antiquewhite                   : #faebd7
aqua                           : #00ffff
aquamarine                     : #7fffd4
azure                          : #f0ffff
beige                          : #f5f5dc
bisque                         : #ffe4c4
black                          : #000000
blanchedalmond                 : #ffebcd
blue                           : #0000ff
blueviolet                     : #8a2be2
brown                          : #a52a2a
burlywood                      : #deb887
cadetblue                      : #5f9ea0
chartreuse                     : #7fff00
chocolate                      : #d2691e
coral                          : #ff7f50
cornflowerblue                 : #6495ed
cornsilk                       : #fff8dc
crimson                        : #dc143c
cyan                           : #00ffff
darkblue                       : #00008b
darkcyan                       : #008b8b
darkgoldenrod                  : #b8860b
darkgray                       : #a9a9a9
darkgrey                       : #a9a9a9
darkgreen                      : #006400
darkkhaki                      : #bdb76b
darkmagenta                    : #8b008b
darkolivegreen                 : #556b2f
darkorange                     : #ff8c00
darkorchid                     : #9932cc
darkred                        : #8b0000
darksalmon                     : #e9967a
darkseagreen                   : #8fbc8f
darkslateblue                  : #483d8b
darkslategray                  : #2f4f4f
darkslategrey                  : #2f4f4f
darkturquoise                  : #00ced1
darkviolet                     : #9400d3
deeppink                       : #ff1493
deepskyblue                    : #00bfff
dimgray                        : #696969
dimgrey                        : #696969
dodgerblue                     : #1e90ff
firebrick                      : #b22222
floralwhite                    : #fffaf0
forestgreen                    : #228b22
fuchsia                        : #ff00ff
gainsboro                      : #dcdcdc
ghostwhite                     : #f8f8ff
gold                           : #ffd700
goldenrod                      : #daa520
gray                           : #808080
grey                           : #808080
green                          : #008000
greenyellow                    : #adff2f
honeydew                       : #f0fff0
hotpink                        : #ff69b4
indianred                      : #cd5c5c
indigo                         : #4b0082
ivory                          : #fffff0
khaki                          : #f0e68c
lavender                       : #e6e6fa
lavenderblush                  : #fff0f5
lawngreen                      : #7cfc00
lemonchiffon                   : #fffacd
lightblue                      : #add8e6
lightcoral                     : #f08080
lightcyan                      : #e0ffff
lightgoldenrodyellow           : #fafad2
lightgreen                     : #90ee90
lightgray                      : #d3d3d3
lightgrey                      : #d3d3d3
lightpink                      : #ffb6c1
lightsalmon                    : #ffa07a
lightseagreen                  : #20b2aa
lightskyblue                   : #87cefa
lightslategray                 : #778899
lightslategrey                 : #778899
lightsteelblue                 : #b0c4de
lightyellow                    : #ffffe0
lime                           : #00ff00
limegreen                      : #32cd32
linen                          : #faf0e6
magenta                        : #ff00ff
maroon                         : #800000
mediumaquamarine               : #66cdaa
mediumblue                     : #0000cd
mediumorchid                   : #ba55d3
mediumpurple                   : #9370db
mediumseagreen                 : #3cb371
mediumslateblue                : #7b68ee
mediumspringgreen              : #00fa9a
mediumturquoise                : #48d1cc
mediumvioletred                : #c71585
midnightblue                   : #191970
mintcream                      : #f5fffa
mistyrose                      : #ffe4e1
moccasin                       : #ffe4b5
navajowhite                    : #ffdead
navy                           : #000080
oldlace                        : #fdf5e6
olive                          : #808000
olivedrab                      : #6b8e23
orange                         : #ffa500
orangered                      : #ff4500
orchid                         : #da70d6
palegoldenrod                  : #eee8aa
palegreen                      : #98fb98
paleturquoise                  : #afeeee
palevioletred                  : #db7093
papayawhip                     : #ffefd5
peachpuff                      : #ffdab9
peru                           : #cd853f
pink                           : #ffc0cb
plum                           : #dda0dd
powderblue                     : #b0e0e6
purple                         : #800080
rebeccapurple                  : #663399
red                            : (255, 0, 0)
rosybrown                      : #bc8f8f
royalblue                      : #4169e1
saddlebrown                    : #8b4513
salmon                         : #fa8072
sandybrown                     : #f4a460
seagreen                       : #2e8b57
seashell                       : #fff5ee
sienna                         : #a0522d
silver                         : #c0c0c0
skyblue                        : #87ceeb
slateblue                      : #6a5acd
slategray                      : #708090
slategrey                      : #708090
snow                           : #fffafa
springgreen                    : #00ff7f
steelblue                      : #4682b4
tan                            : #d2b48c
teal                           : #008080
thistle                        : #d8bfd8
tomato                         : #ff6347
turquoise                      : #40e0d0
violet                         : #ee82ee
wheat                          : #f5deb3
white                          : (255, 255, 255)
whitesmoke                     : #f5f5f5
yellow                         : #ffff00
yellowgreen                    : #9acd32
'''



def crop_data_parts(psp_fileName, image_filename, mask_filename, save_dir1, save_dir2,win_size):
    fp = open(psp_fileName)
    classRects = []

    colors = {}
    classRects = []
    curColor = ""
    rect_flag = True
    for k, txt in enumerate(fp):
        if k<2:
            continue
        zz, t = txt.split('=')
        if t.find('Pline') >= 0:
            x2y = []
            rect_flag = False
            continue
        if t.find('Rectangle') >= 0:
            x2y = []
            rect_flag = True
            continue
        if t.find('Pen') >= 0:
            p = t.split(',')
            curColor = p[2]
            if p[2] not in colors:
                colors.update({p[2]: len(colors)})
        t = t.split(' ')
        if t[0] == '':
            if rect_flag:
                x2y.insert(1, (x2y[0][0], x2y[1][1]))
                x2y.append((x2y[2][0], x2y[0][1]))
            idx = colors.get(curColor)
            if idx ==  len(classRects):
                classRects.append([x2y])
            else:
                classRects[idx].append(x2y)
            continue
        x2y.append((int(t[0]), int(t[1])))
    fp.close()
    nmb_fragm = 0
    im = Image.open(image_filename)
    mask = Image.open(mask_filename)
    bound = math.ceil(math.sqrt(2) * win_size - win_size)
    for j in range(len(classRects[0])):
        x0 = classRects[0][j][0][0]
        y0 = classRects[0][j][0][1]
        x_len = classRects[0][j][1][1]-classRects[0][j][0][1]
        y_len = classRects[0][j][2][0]-classRects[0][j][1][0]

        im2 = im.crop((x0-bound, y0-bound, x0 +x_len+bound, y0+bound+ y_len))
        im3 = mask.crop((x0-bound, y0-bound, x0 +bound + x_len, y0+bound + y_len))

        im2.save(save_dir1+"/+++data_fragm{:05d}.BMP".format(nmb_fragm))
        im3.save(save_dir2 + "/+++data_fragm{:05d}.BMP".format(nmb_fragm))
        nmb_fragm += 1
    im.close()
    mask.close()


def create_class_masks(filename, save_dir, class_numb):
    picture = Image.open(filename)
    for l in range(class_numb):

        width_p, height_p = picture.size
        img_accumulated = Image.new('RGB', (width_p, height_p), color=0)
        mass = np.array(picture)
        print(mass.shape, width_p, height_p)
        for i in range(height_p):
            for j in range(width_p):
                if mass[i, j, 0] == t_color_codes[l][0] and mass[i, j, 1] == t_color_codes[l][1] and\
                        mass[i, j, 2] == t_color_codes[l][2]:
                    img_accumulated.putpixel((j, i), (0, 0, 255))
        img_accumulated.save(save_dir + "mask_class_" + str(l) + ".BMP")
        img_accumulated.close()
    picture.close()

def make_split_augmentation(dir_elems1, dir_elems2, save_dir1, save_dir2, win_size, step_len):
    datas = []
    masks = []
    for file in os.listdir(dir_elems1):
        if file.endswith(".BMP"): datas.append(file)
    for file in os.listdir(dir_elems2):
        if file.endswith(".BMP"): masks.append(file)
    bound = math.ceil(math.sqrt(2) * win_size - win_size)
    def crop_parts(dir_elems, save_dir, filelist):
        nmb_fragm = 0
        for file in filelist:
            im = Image.open(dir_elems + "\\" + file)
            x0,y0 = bound,bound
            x_len, y_len = im.size
            x_len -= 2*bound+win_size
            y_len -= 2*bound +win_size
            x_amount = int(x_len / step_len)
            y_amount = int(y_len / step_len)
            for k in range(x_amount):
                for l in range(y_amount):

                    im2 = im.crop((x0 + k * step_len - bound, y0 + l * step_len - bound,
                                   x0 + k * step_len + bound + win_size, y0 + l * step_len + bound + win_size))
                    imfnam = save_dir + "/+++fragm{:05d}.BMP".format(nmb_fragm)
                    nmb_fragm += 1;
                    im2.save(imfnam)
            im.close()

    crop_parts(dir_elems1, save_dir1, datas)
    crop_parts(dir_elems2, save_dir2, masks)



    # rjkbxtcndj преобразований , выполняемых над изображением
    transAmount = 6;
    dlin = 0

    def load_CNN_train_augment(dir_name1,dir_name2,win_size):
        ll1 = []  # пустой список имен входных файлов JPG
        ll2 = []
        for file in os.listdir(dir_name1):
            if file.endswith(".BMP"): ll1.append(file)
        for file in os.listdir(dir_name2):
            if file.endswith(".BMP"): ll2.append(file)
        dlin = len(ll1)
        dlin0 = int(len(ll1) / 2)  # будем пропускать нечетные элементы
        train_x = np.zeros((dlin * transAmount, win_size, win_size, 3), dtype='float32')
        train_y = np.zeros((dlin * transAmount, win_size, win_size, 3), dtype='int')
        train_z = np.zeros((dlin * transAmount, 1), dtype='float32')
        # dlin = dlin * transAmount
        k = 0
        for file in ll1:
            im = Image.open(dir_name1 + "\\" + file)
            im2 = Image.open(dir_name2 + "\\" + file)
            train_x[transAmount * k] = np.array(im.crop((bound, bound, bound + win_size, bound + win_size))) * 1. / 255.
            train_x[transAmount * k + 1] = np.array(
                im.rotate(45).crop((bound, bound, bound + win_size, bound + win_size))) * 1. / 255.  # print(b0.shape)

            train_y[transAmount * k] = np.array(im2.crop((bound, bound, bound + win_size, bound + win_size)))
            train_y[transAmount * k + 1] = np.array(
                im2.rotate(45).crop((bound, bound, bound + win_size, bound + win_size)))
            shift0 = random.randint(0, 6)
            shift1 = random.randint(0, 6)
            shift2 = random.randint(0, 6)
            shift3 = random.randint(0, 6)
            if shift0 == 0 and shift1 == 0:
                shift0 += 2
            if shift2 == 0 or shift2 == 3:
                shift2 += 1
            if shift3 == 0 or shift3 == 3:
                shift3 += 1
            train_x[transAmount * k + 2] = np.array(
                im.crop((shift0, 0, shift0 + win_size, win_size))) * 1. / 255.  # print(b0.shape)

            train_x[transAmount * k + 3] = np.array(
                im.crop((0, shift1, win_size, shift1 + win_size))) * 1. / 255.  # print(b0.shape)
            train_x[transAmount * k + 4] = np.array(
                im.rotate(90).crop((shift2, shift3, shift2 + win_size, shift3 + win_size))) * 1. / 255.
            train_x[transAmount * k + 5] = np.array(
                im.rotate(180).crop((bound, bound, bound + win_size, bound + win_size))) * 1. / 255.
            # train_x[transAmount * k + 4] = np.array(im.crop((0,8,13,21)))*1./255.

            train_y[transAmount * k + 2] = np.array(
                im2.crop((shift0, 0, shift0 + win_size, win_size)))

            train_y[transAmount * k + 3] = np.array(
                im2.crop((0, shift1, win_size, shift1 + win_size)))
            train_y[transAmount * k + 4] = np.array(
                im2.rotate(90).crop((shift2, shift3, shift2 + win_size, shift3 + win_size)))
            train_y[transAmount * k + 5] = np.array(
                im2.rotate(180).crop((bound, bound, bound + win_size, bound + win_size)))
            # train_x[transAmount * k + 4] = np.array(im.crop((0,8,13,21)))*1./255.

            im.close()
            k += 1
            if k == dlin: break
        #temp = []
        #for i in range(len(train_z)):
            #temp.append(to_categorical(train_z[i], num_classes=class_amount))
        #train_y = np.array(temp)
        return train_x, train_y

    def one_hot_encoding(train_y, class_amount):
        shape_y = np.shape(train_y)
        train_z = np.zeros((shape_y[0], shape_y[1], shape_y[2], class_amount), dtype='int')
        for l in range(shape_y[0]):
            for i in range(shape_y[1]):
                for j in range(shape_y[2]):
                    #print(l,i,j)
                    list_diffs = np.array(t_color_codes) -train_y[l, i, j, :]
                    list_diffs = list_diffs[:,0]**2+list_diffs[:,1]**2+list_diffs[:,2]**2
                    #np.argmin(list_diffs)
                    train_z[l,i,j,:] = np.eye(class_amount)[np.array(np.argmin(list_diffs))]
        return train_z


    train_x1, train_y1 = load_CNN_train_augment(save_dir1, save_dir2, win_size)
    train_y1 = one_hot_encoding(train_y1, 4)


#crop_data_parts("D:/image_data/__FR1_modified.jpg._PSP", "D:/image_data/__FR1_modified.jpg", "D:/image_data/new_mask_class_0.BMP",\
               #"D:/image_data/parts_with_bounds/data","D:/image_data/parts_with_bounds/mask", 16)



#create_class_masks("D:/image_data/new_mask_class_0.BMP", "D:/image_data/parts_with_bounds/mask_classes/", 4)
make_split_augmentation("D:/image_data/parts_with_bounds/data", "D:/image_data/parts_with_bounds/mask",\
                        "D:/image_data/parts_with_bounds/data_crop", "D:/image_data/parts_with_bounds/mask_crop", 16, 4)