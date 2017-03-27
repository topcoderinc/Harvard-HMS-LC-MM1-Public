import cPickle as pickle
import os
import argparse
import sys
from bunch import Bunch
import math
import cv2
import timeit
import os
import random

import tensorflow as tf
import numpy as np
import pandas as pd
#from PIL import Image

from utils import select_gpu

LAPTOP = 'MAREK_LAPTOP' in os.environ
#EC = 'USING_EC2' in os.environ
#TRAIN_DIR = 'new-data/'
CROP_SIZE = (0, 0)
RECTANGLES = -1
PIECES = -1
MODE_HEATMAP = 1
MODE_RECTANGLES = 2
MODE_RECTANGLES_FAKE = 3
MODE = 0
FLIP_VERTICAL = 0
FLIP_HORIZONTAL = 0
BAND8 = 0

def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name='weight')

def bias_variable(shape, bias):
    initial = tf.constant(bias, shape=shape)
    return tf.Variable(initial, name='bias')

def conv2d(x, W, stride=1):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def myconv(signal, size, channels, filters, stddev, bias, stride=1, scope='conv'):
    with tf.variable_scope(scope):
        dev = stddev if stddev is not None else math.sqrt(1.0 / (size * size * channels))
        print 'std dev', dev
        W_conv = weight_variable([size, size, channels, filters], stddev=dev)
        b_conv = bias_variable([filters], bias=bias)
    return conv2d(signal, W_conv, stride=stride) + b_conv

def mydilatedconv(signal, size, rate, channels, filters, stddev, bias, scope='dilconv'):
    with tf.variable_scope(scope):
        dev = stddev if stddev is not None else math.sqrt(1.0 / (size * size * channels))
        print 'std dev', dev
        W_conv = weight_variable([size, size, channels, filters], stddev=dev)
        b_conv = bias_variable([filters], bias=bias)
    return tf.nn.atrous_conv2d(signal, W_conv, rate=rate, padding='SAME') + b_conv

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def batch_norm(signal, phase_train, scope='batch_norm', scale=True, decay=0.98):
    """
    Batch normalization
    Args:
       signal:      Tensor, 4D BHWD input maps (or any other arbitrary shape that make sense)
       phase_train: boolean, true indicates training phase, false for test time (placeholder would be a good choice)
       scope:       string, variable scope
       scale:       boolean, whether to allow output scaling
    Return:
       normed:      batch-normalized signal
    """

    with tf.variable_scope(scope):
        n_out = int(signal.get_shape()[-1])  # depth of input signal (value of the last dimension)

        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=scale)

        ema = tf.train.ExponentialMovingAverage(decay=decay)

        batch_mean, batch_variance = tf.nn.moments(signal, range(signal.get_shape().ndims - 1), name='moments')

        def mean_and_var_for_train():
            ema_apply_op = ema.apply([batch_mean, batch_variance])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        def mean_and_var_for_test():
            ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_variance)
            return ema_mean, ema_var

        mean, var = tf.cond(phase_train, mean_and_var_for_train, mean_and_var_for_test)
        normed = tf.nn.batch_normalization(signal, mean, var, beta, gamma, 1e-5)
        return normed

#returns logits
def construct_dilated(signal, input_channels, arch, stddev, bias, final_bn=True, use_batch_norm=True, phase_train=None):
    image_size = CROP_SIZE
    channels = input_channels

    for idx, p in enumerate(arch):
        if len(p) == 2:
            filters, rate = p
            size = 3
            pool = 0
        else:
            assert(len(p) == 4)
            filters, rate, size, pool = p

        signal = mydilatedconv(signal, size=size, rate=rate, stddev=stddev, bias=bias, channels=channels, filters=filters)
        if idx < len(arch)-1:
            if use_batch_norm:
                signal = batch_norm(signal, phase_train, scale=True)
            signal = tf.nn.relu(signal)
            if pool:
                signal = max_pool_2x2(signal)
                image_size = (image_size + 1) / 2
        elif final_bn:
            signal = batch_norm(signal, phase_train, scale=True)
        channels = filters
        print 'dil conv layer with {} filters, rate {}, size {}, image_size'.format(channels, rate, size, image_size)
        print signal

    assert(image_size == FINAL_SIZE)
    return signal

#initially signal has shape [batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], input_channels]
def construct_convnet(signal, input_channels, arch, ending_size_dict, stddev, bias, rectangles, use_batch_norm=True, use_dropout=False, keep_prob=None, phase_train=None, ff=1):
    image_size = (CROP_SIZE, CROP_SIZE)
    channels = input_channels

    for idx, (filters, pool, size) in enumerate(arch):
        if use_dropout:
            print 'dropout applied'
            signal = tf.nn.dropout(signal, keep_prob)
        signal = myconv(signal, size=size, stddev=stddev, bias=bias, channels=channels, filters=filters)
        if use_batch_norm:
            signal = batch_norm(signal, phase_train, scale=True)
        signal = tf.nn.relu(signal)
        if pool:
            signal = max_pool_2x2(signal)
            image_size = ((image_size[0] + 1) / 2, (image_size[1] + 1) / 2)
        channels = filters
        print 'conv layer with {} filters, output image size {}'.format(channels, image_size)

    signal2 = myconv(signal, size=ending_size_dict['offset'], stddev=stddev, bias=bias, channels=channels, filters=5*rectangles)
    signal = myconv(signal, size=ending_size_dict['main'], stddev=stddev, bias=bias, channels=channels, filters=rectangles)
    print 'image_size after convolutions', image_size
    return signal, signal2, image_size

def load_valid_heatmaps(validlist):
    dict = {}
    namelist = extract_names(validlist)
    #print len(validlist), validlist[:2]
    #print len(namelist), namelist[:2]
    #print len(zip(namelist, validlist))
    first = True
    for (name, (_, _, _, heatmap_path, _, _)) in zip(namelist, validlist):
        heatmap = cv2.imdecode(np.array(pickle.load(open(heatmap_path))[0]), -1).astype(np.float32) / 255.0
        dict[name] = cv2.resize(heatmap, (FINAL_SIZE, FINAL_SIZE))
        if first:
            first = False
            print 'valid_dict', dict
    return dict

def myrotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def reverse_augmentation(pred, aug_data):
   dx, dy, offx, offy, angle = aug_data
   a = AUG_SIZE/2
   size = CROP_SIZE + AUG_SIZE
   mysize = (CROP_SIZE + dx, CROP_SIZE + dy)
   assert(pred.shape[0] == FINAL_SIZE)
   assert(pred.shape[1] == FINAL_SIZE)

   pred = cv2.resize(pred, (CROP_SIZE, CROP_SIZE))
   img = np.zeros((size, size, pred.shape[2]), dtype=np.float32)
   img[a:a+CROP_SIZE, a:a+CROP_SIZE, :] = pred
   img = img[offx:offx+mysize[0], offy:offy+mysize[1], :]
   img = cv2.resize(img, (CROP_SIZE, CROP_SIZE))
   if angle:
       img = myrotate(img, -angle)

   img = cv2.resize(img, (FINAL_SIZE, FINAL_SIZE))
   return img

def my_read_image(x):
    dx = random.randint(-AUG_SIZE, AUG_SIZE)
    dy = random.randint(-AUG_SIZE, AUG_SIZE)
    angle = random.randint(-ROTATE, ROTATE)
    mysize = (CROP_SIZE + dx, CROP_SIZE + dy)

    img_path, img_path0, img_path2, heatmap_path, mask, weight = x
    weight = float(weight)
    imgs = []
    for path in [img_path, img_path0, img_path2]:
        img = cv2.imread(path,-1).astype(np.float32)
        if angle:
            img = myrotate(img, angle)
        img = cv2.resize(img, (mysize[1], mysize[0]))
        #img = img.reshape(IMAGE_SIZE+(1,))
        imgs.append(img)
    img = np.stack(imgs, axis=2)
    if AUG_COLORS == 1:
        mean = np.mean(img)
        img = (img-mean) * random.uniform(0.97, 1.03) + mean + random.uniform(-100,100)
    #print 'img.shape', img.shape


    if heatmap_path != 'None' and heatmap_path != 'NONE':
        jpgs = [cv2.imdecode(np.array(buf), -1).astype(np.float32) / 255.0 for buf in pickle.load(open(heatmap_path, 'rb'))[:OUT_CHANNELS]]
        heatmap = np.stack(jpgs, axis=2)

        if angle:
            heatmap = myrotate(heatmap, angle)
        heatmap = cv2.resize(heatmap, (mysize[1], mysize[0]))

        #if heatmap_path[-4:] != '.jpg':
        #    heatmap_path += '.jpg'
        #heatmap = cv2.imread(heatmap_path, -1).astype(np.float32) / 255.0
        #heatmap = cv2.resize(heatmap, mysize)
        #heatmap = heatmap.reshape(mysize+(1,))
       # print 'heat max', np.max(heatmap)
    else:
        heatmap = np.zeros(mysize+(OUT_CHANNELS,), dtype=np.float32)

    size = CROP_SIZE + AUG_SIZE
    newheat = np.zeros((size, size, OUT_CHANNELS), dtype=np.float32)
    newimg = np.zeros((size, size, NUM_CHANNELS), dtype=np.float32)
    offx = random.randint(0, size - mysize[0])
    offy = random.randint(0, size - mysize[1])
    newheat[offx:offx+mysize[0], offy:offy+mysize[1], :] = heatmap
    newimg[offx:offx+mysize[0], offy:offy+mysize[1], :] = img

    a = AUG_SIZE/2
    heatmap = newheat[a:a+CROP_SIZE, a:a+CROP_SIZE, :]
    img = newimg[a:a+CROP_SIZE, a:a+CROP_SIZE, :]

    heatmap = cv2.resize(heatmap, (FINAL_SIZE, FINAL_SIZE))

    #print 'shapes', img.shape, heatmap.shape
    #print 'types', img.dtype, heatmap.dtype
    if mask == 'NONE':
        mask = '000000000000'
    mask = [mask[i] for i in range(len(mask))]
    assert(len(mask) == 12)
    mask = np.array(mask[:OUT_CHANNELS], dtype=np.float32)
    return img, heatmap, mask, np.array([weight], dtype=np.float32), [dx, dy, offx, offy, angle]

def save_images(name, im):
    print 'name', name
    print 'im.shape', im.shape
    for i in range(im.shape[0]):
        cv2.imwrite('batches/'+name+'_'+str(i)+'.jpg', im[i][:, :, :3]*255)
        if im[i].shape[2] > 3:
            cv2.imwrite('batches/'+name+'_'+str(i)+'_rest.jpg', im[i][:, :, 3:6]*255)

def estimate_mean(sess, single_batch, sampled_batches=20):
    mean = None
    meanheat = None
    for batchid in range(sampled_batches):
        names, im, heat, _, _, _ = sess.run(single_batch)
        #im, heat = sess.run(single_batch)
        #save_images('im_'+str(batchid), im)
        #save_images('heat_'+str(batchid), heat)
     #   print 'has batch', names
        im = np.mean(im, axis=(0,1,2))
        h = np.mean(heat, axis=(0,1,2))
        if mean is None:
            mean = im
            meanheat = h
        else:
            mean += im
            meanheat += h
    mean /= sampled_batches
    meanheat /= sampled_batches
    meanheat = np.log(meanheat) * meanheat + (1.0 - meanheat) * np.log(meanheat)
    print mean
    print 'meanheat', meanheat

    var = None
    for _ in range(sampled_batches):
        _, im, _, _, _, _ = sess.run(single_batch)
        print 'has batch'
        v = np.mean((im - mean)**2, axis=(0,1,2))
        if var is None:
            var = v
        else:
            var += v
    var /= sampled_batches
    print var
    return mean, np.sqrt(var)

def tf_batch(file_paths, batch_size, shuffle=True):
    print file_paths[0]
    print 'file_paths_len', len(file_paths)

    input_queue = tf.train.slice_input_producer([file_paths], shuffle=shuffle, capacity=256)
    image, heatmap, mask, weight, aug_data = tf.py_func(my_read_image, [input_queue[0]], [tf.float32, tf.float32, tf.float32, tf.float32, tf.int64])

    #single_batch = tf.train.batch([input_queue[0], image, heatmap], batch_size=batch_size,
    #                              num_threads=2 if LAPTOP else 6, capacity=6*batch_size,
    #                              shapes=[(2), (IMAGE_CROP[0], IMAGE_CROP[1], NUM_CHANNELS), (IMAGE_CROP[0], IMAGE_CROP[1], OUT_CHANNELS)])
    shape_list = [(6), (CROP_SIZE, CROP_SIZE, NUM_CHANNELS), (FINAL_SIZE, FINAL_SIZE, OUT_CHANNELS), (OUT_CHANNELS), (1), (5)]
    print 'shape_list', shape_list
    single_batch = tf.train.batch([input_queue[0], image, heatmap, mask, weight, aug_data], batch_size=batch_size,
                                  num_threads=2 if LAPTOP else 6, capacity=6*batch_size,
                                  shapes=shape_list)
    return single_batch

def make_shift(l):
    res = []
    for s in l:
        v = s.split(',')
        shifts = [0, 4]
        #shifts = [0]
        scales = [0, 2, 4, 6]
        #shifts = [0]
        #shifts = [25]
        for s in scales:
            for i in shifts:
                for j in shifts:
                    suf = ''
                    for x in [s, i, j]:
                        suf += '_' + chr(ord('a')+x)
                    res.append(','.join([v[0]+suf]+v[1:]))
    return res

def make_train_list(patients, dict, dir, dict_mask, dict_weight, has_heat=True):
    res = []
    for p in patients:
        assert(p in dict)
        scans = dict[p]
        for i in range(1,len(scans)-1):
            s0 = scans[i-1]
            s = scans[i]
            s2 = scans[i+1]
            image_path0 = '/'.join([dir, p, 'pngs', s0+'.png'])
            image_path = '/'.join([dir, p, 'pngs', s+'.png'])
            image_path2 = '/'.join([dir, p, 'pngs', s2+'.png'])
            heatmap_path = '/'.join([HEATMAP_DIR, p+'_'+s+'.p']) if has_heat else 'NONE'
            mask = dict_mask[(p, s)] if dict_mask else 'NONE'
            weight = dict_weight[p] if dict_weight else '0.0'
            res.append((image_path, image_path0, image_path2, heatmap_path, mask, str(weight)))
    return res

def read_dict(filename):
    l = open(filename).readlines()
    dict = {}
    for s in l:
        v = s.split(',')
        patient = v[0]
        scanid = v[1]
        if patient not in dict:
            dict[patient] = []
        dict[patient].append(scanid)
    return dict

def read_dict_mask():
    dict = {}
    for s in open('12-masks.csv').readlines():
        v = s.split(',')
        patient = v[0]
        scanid = v[1]
        mask = ''.join(v[2:])
        dict[(patient,scanid)] = mask[:-1]
    return dict

def read_dict_weight():
    dict = {}
    for s in open('weights.csv').readlines():
        v = s.split(',')
        patient = v[0]
        weight = float(v[-1])
        dict[patient] = weight
    sum = 0.
    for p in dict.keys():
        sum += dict[p]
    sum /= len(dict.keys())

    for p in dict.keys():
        dict[p] /= sum

    print 'dict_weights', dict
    return dict


def read_csv(valid_seed, reverse_order):
    dict = read_dict('scans_all.csv')
    dict_test = read_dict('scans_test.csv')
    dict_mask = read_dict_mask()
    dict_weight = read_dict_weight()

    patients = sorted(dict.keys())
    print patients, len(patients)

    patients_test = sorted(dict_test.keys())
    print patients_test, len(patients_test)

    random.seed(valid_seed)
    random.shuffle(patients)
    if reverse_order:
        patients.reverse()
    valid_size = len(patients) / 5
    adir = 'example_extracted'
    bdir = 'provisional_extracted_no_gt'
    return make_train_list(patients[:valid_size], dict, adir, dict_mask, dict_weight), \
           make_train_list(patients[valid_size:], dict, adir, dict_mask, dict_weight), \
           make_train_list(patients_test, dict_test, bdir, dict_weight=None, dict_mask=None, has_heat=False)

def dump_logits(logits, sig2, names, threshold, nameset):
    logits = logits.reshape((-1,PIECES,PIECES,RECTANGLES))
    sig2 = sig2.reshape((-1,PIECES,PIECES,RECTANGLES,5))
    print logits.shape, threshold
    assert(logits.shape[1] == PIECES and logits.shape[2] == PIECES)
    assert(logits.shape[3] == RECTANGLES)
    previ = -1
    a, b, c, d = np.where(logits >= threshold)
    for i, px, py, r in zip(a, b, c, d):
        if i in nameset:
            continue
        if i != previ:
            if previ >= 0:
                f.close()
            previ = i
            f = open('predictions/'+str(names[i]), 'w')
            f2 = open('predictions/'+str(names[i])+'.meta', 'w')
        print >>f, logits[i][px][py][r], px, py, r
        print >>f2, sig2[i][px][py][r][0], sig2[i][px][py][r][1], sig2[i][px][py][r][2],\
            sig2[i][px][py][r][3], sig2[i][px][py][r][4], px, py, r
    if previ != -1:
        f.close()
        f2.close()

    #for i in range(logits.shape[0]):
    #    f = open('predictions/'+str(names[i]), 'w')
    #    for px in range(PIECES):
    #        for py in range(PIECES):
    #            for r in range(RECTANGLES):
    #                if logits[i][px][py][r] > threshold:
    #                    print >>f, logits[i][px][py][r], px, py, r
    #    f.close()

def extract_names(names):
    return map(lambda (x,_i,_j,_k,_l, _m) : x.split('/')[1]+'_'+x.split('/')[3][:-4], names)

def calc_mask(logits, labels, nonzeros, nonzeros_mult):
    rect = logits.shape[1]
    assert(logits.shape == labels.shape)
    print 'logits and labels, shapes:', logits.shape, labels.shape
    print 'nonzeros', nonzeros
    res = np.array(labels)
    res2 = np.zeros(labels.shape+(5,), dtype=np.int32)
    a, b = np.where(labels > 0)
    #print a, b
    for i in range(5):
        res2[a, b, [i]] = 1
    print 'res2.shape', res2.shape
    positions = np.argpartition(logits.reshape([-1]), -int(nonzeros_mult*nonzeros))[-int(nonzeros_mult*nonzeros):] #TODO constant
    print 'positions', positions, len(positions)
    a = positions / rect
    b = positions % rect
    res[a, b] = 1
    print 'mask nonzeros', np.sum(res)
    hit = np.sum(labels[a, b])
    print 'labels hit', hit, 'nonzeros', nonzeros
    return res, res2.reshape([-1,rect*5]), float(hit) / (nonzeros+0.)

def train(args):
    global PIECES, RECTANGLES, MODE, NUM_CHANNELS, OUT_CHANNELS, CROP_SIZE, ROTATE, AUG_SIZE, FLIP_VERTICAL, FLIP_HORIZONTAL, BAND8, HEATMAP_DIR, FINAL_SIZE
    global AUG_COLORS

    AUG_COLORS = args.aug_colors
    HEATMAP_DIR = args.heatmap_dir
    AUG_SIZE = args.aug_size
    ROTATE = args.rotate
    CROP_SIZE = args.crop
    FINAL_SIZE = args.final
    NUM_CHANNELS = 3
    OUT_CHANNELS = args.out_channels

    device = '/cpu:0' if args.cpu else '/gpu:{idx}'.format(idx=select_gpu())
    print 'device', device
    valid_list, train_list, test_list = read_csv(args.valid_seed, args.reverse_order)
    print len(train_list), train_list[:10], len(valid_list), valid_list[:10]
    print len(test_list), test_list[:10]

    with tf.device(device):
        x = tf.placeholder(tf.float32, shape=[None, CROP_SIZE, CROP_SIZE, NUM_CHANNELS], name='x')
        y_heat = tf.placeholder(tf.float32, shape=[None, FINAL_SIZE, FINAL_SIZE, OUT_CHANNELS], name='y_heat')
        mask = tf.placeholder(tf.float32, shape=[None, OUT_CHANNELS], name='mask')
        coefficients = tf.placeholder(tf.float32, shape=[OUT_CHANNELS], name='coefficients')

        signal = x
        tf.summary.image('input', x, max_outputs=10)
        for i in range(OUT_CHANNELS):
            tf.summary.image('heatmap_'+str(i), tf.slice(y_heat, (0, 0, 0, i), (-1, -1, -1, 1)), max_outputs=5)

        keep_prob = tf.placeholder("float")
        learning_rate = tf.placeholder(tf.float32)
        phase_train = tf.placeholder(tf.bool)

        if args.dilarch == 0:
            #front-end
            dilarch = [(8, 1, 3, 0), (8, 1, 3, 1),
                       (16, 1, 3, 0), (16, 1, 3, 1),
                       (32, 1, 3, 0), (32, 1, 3, 0),
                       (64, 2, 3, 0), (64, 2, 3, 0),
                       (64, 4, 3, 0), (64, 4, 3, 0),
                       (64, 8, 3, 0), (64, 8, 3, 0),
                       (64, 16, 3, 0), (64, 16, 3, 0),
                       (64, 4, 3, 0), (64, 4, 3, 0)] + args.num_dense * [(args.neuron_dense, 1, 1, 0)] + [(OUT_CHANNELS, 1, 1, 0)]
            dilarch = map(lambda (a,b,c,d): (int(args.arch_multiplier*a), b, c, d), dilarch[:-1])+dilarch[-1:]
            if args.filters_bound:
                dilarch = map(lambda (a,b,c,d): (min(a, args.filters_bound), b, c, d), dilarch[:-1])+dilarch[-1:]
        elif args.dilarch == 1:
            dilarch = [(10, 1), (10, 1), (10, 2), (10, 4), (10, 8), (10, 16), (10, 32), (10, 16), (10, 4), (OUT_CHANNELS, 1)]
            dilarch = map(lambda (a,b): (int(args.arch_multiplier*a), b), dilarch[:-1])+dilarch[-1:]
        else:
            assert(0)
        print 'dilarch', dilarch

        logits = construct_dilated(signal, input_channels=NUM_CHANNELS, arch=dilarch, stddev=args.conv_stddev,
                                   bias=args.conv_bias, use_batch_norm=args.batch_norm,
                                   final_bn=args.final_bn, phase_train=phase_train)
        sigmoid = tf.nn.sigmoid(logits)
        logloss = tf.nn.sigmoid_cross_entropy_with_logits(logits, y_heat)
        mean_logloss = tf.reduce_mean(logloss, axis=[1, 2])
        print 'mean_logloss', mean_logloss
        row_objective = tf.reduce_sum(mean_logloss * mask, axis=0)
        print 'row_objective', row_objective
        objective = tf.reduce_sum(row_objective * coefficients)


        num_batches_train = 400 #len(train_list) / args.batch_size
        num_batches_valid = len(valid_list) / args.batch_size
        print 'num_batches train {}, valid {}'.format(num_batches_train, num_batches_valid)

        #global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(args.lr, global_step,
        #                                           num_batches_train, args.epoch_decay, staircase=False)
        if args.optimizer == 'adam':
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(objective)
        elif args.optimizer == 'momentum':
            train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(objective)
        else:
            assert(0)

        #for var in tf.all_variables():
        #    print var.name

        train_batch = tf_batch(train_list, batch_size=args.batch_size)
        valid_batch = tf_batch(valid_list, batch_size=args.batch_size)
        test_batch = tf_batch(test_list, batch_size=args.batch_size)

        saver = tf.train.Saver(max_to_keep=100)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            merged = tf.summary.merge_all()
            tensorboard_writer = tf.summary.FileWriter('logs', sess.graph)

            print 'session started'
            sess.run(tf.global_variables_initializer(), feed_dict={phase_train: True})
            print 'variables initialized'
            threads = tf.train.start_queue_runners(sess, coord=tf.train.Coordinator())
            print 'queues started'
            if args.means:
                mean, stddev = pickle.load(open(args.means))
                mean = np.array(mean)
                stddev = np.array(stddev)
            else:
                mean, stddev = estimate_mean(sess, train_batch, sampled_batches=5)
            print 'mean estimated, shape={}, flattened={}'.format(mean.shape, mean)
            print 'stddev estimated, shape={}, flattened={}'.format(stddev.shape, stddev)

            if args.means_store:
                f = open(args.means_store, 'w')
                pickle.dump((mean, stddev), f)
                f.close()

            epoch = 0
            if args.restore_path:
                saver.restore(sess, args.restore_path)

            if args.predict:
                #pass_list = [(train_batch, len(train_list)-13*args.batch_size, 'train', True),
                #             (train_batch, len(train_list), 'train', False),
                #             (valid_batch, len(valid_list), 'val', False),
                #             (test_batch, len(test_list), 'test', False)]
                pass_list = []
                assert(len(args.predict) == 3)
                pass_list = []

                if args.predict[0] == '1':
                    pass_list.append((train_batch, train_list[:200 * 32], 'train', True))

                if args.predict[1] == '1':
                    pass_list.append((valid_batch, valid_list, 'val', False))
                    valid_dict = load_valid_heatmaps(valid_list)

                if args.predict[2] == '1':
                    pass_list.append((test_batch, test_list, 'test', False))

                for (batch_gen, namelist, pref, pt) in pass_list:
                    size = len(namelist)
                    print size
                    print namelist[:10]
                    namelist = extract_names(namelist)
                    print namelist[:10]
                    nameset = {}
                    pred_dict = {}
                    tot_cnt = 0
                    for name in namelist:
                        nameset[name] = 0

                    while tot_cnt < size * (1 if pref == 'train' else args.pred_iter):
                        print 'tot_cnt', tot_cnt
                        names, batch_x, _, _, _, aug_data = sess.run(batch_gen)
                        batch_x -= mean
                        batch_x /= stddev
                        print 'names', names[:10], len(names)
                        names = extract_names(names)
                        print 'names', names[:10], len(names)

                        print 'shape', batch_x.shape
                        [heat_pred] = sess.run([sigmoid], feed_dict={x: batch_x, phase_train: pt}) #TODO check false
                        print 'heat_pred.shape', heat_pred.shape
                        if tot_cnt == 0:
                            print 'aug_data', aug_data

                        for i in range(args.batch_size):
                            name = names[i]
                            if pref == 'train' or nameset[name] < args.pred_iter:
                                if pref != 'train':
                                    nameset[name] += 1
                                tot_cnt += 1
                                if not pt:
                                    newheat = reverse_augmentation(heat_pred[i, :, :, :], aug_data[i])
                                    # newheat =
                                    #[dx] = aug_data[i]
                                    #if dx > 0:
                                    #    s = CROP_SIZE + 2 * dx
                                    #    newheat = np.zeros((s, s, OUT_CHANNELS), dtype=np.float32)
                                    #    newheat[dx:CROP_SIZE+dx, dx:CROP_SIZE+dx, :] = heat_pred[i, :, :, :]
                                    #else:
                                    #    dx = -dx
                                    #    newheat = heat_pred[i, dx:CROP_SIZE-dx, dx:CROP_SIZE-dx, :]
                                    #newheat = heat_pred[i, :, :, :]

                                    #newheat = cv2.resize(newheat, (CROP_SIZE, CROP_SIZE))
                                    newheat = newheat[:, :, 0:1] #storing only the cancer
                                    if pref != 'train':
                                      if nameset[name] == 1:
                                          pred_dict[name] = newheat
                                      else:
                                          pred_dict[name] += newheat

                    if pref == 'train':
                        save_path = saver.save(sess, args.models_dir + '/final.ckpt')
                        print ('Model saved in file: %s' % save_path)

                    if pref != 'train':
                        losses = []
                        for name in namelist:
                            newheat = pred_dict[name] / nameset[name]
                            if pref == 'val':
                                assert(name in valid_dict)
                                loss = valid_dict[name] * np.log(newheat[:, :, 0]+1e-6) +\
                                       (1.0 - valid_dict[name]) * (np.log(1.0-newheat[:, :, 0]+1e-6))
                                loss = np.mean(loss)
                                print 'loss', loss
                                losses.append(loss)
                            #for c in range(OUT_CHANNELS):
                            for c in range(1):
                                curpred = newheat[:, :, c]
                                print 'mean', np.mean(curpred), 'max', np.max(curpred)
                                curpred *= 255.0
                                print 'mean', np.mean(curpred), 'max', np.max(curpred)
                                img = np.asarray(curpred, dtype=float)
                                print 'mean', np.mean(img), 'max', np.max(img)
                                img = cv2.resize(img, (512, 512))
                                cv2.imwrite('predictions/'+pref+'/'+name+'_'+str(c)+'.jpg', img)
                                #cv2.imwrite('predictions/'+pref+'/'+name+'_'+str(c)+'.bmp', img)
                                #TODO
                        print 'ave loss', np.mean(losses)
                    if pref == 'val':
                        valid_dict = {}
                return


            epoch_lr = args.lr
            while epoch < args.num_epochs:
                epoch += 1
                print 'epoch {} learning rate {}'.format(epoch, epoch_lr)
                epoch_start_time = timeit.default_timer()
                train_time, load_time, valid_time, dump_time = 0, 0, 0, 0

                pass_list = [([objective, row_objective, train_step], num_batches_train, train_batch, True),
                             ([objective, row_objective], num_batches_valid, valid_batch, False)]

                for compute_list, num_batches, batch_gen, pt in pass_list:
                    load_time = 0
                    main_time = 0
                    pass_start = timeit.default_timer()
                    results = []
                    means = []
                    for i in range(num_batches):
                        start = timeit.default_timer()
                        _, batch_x, batch_heat, batch_mask, batch_weight, _ = sess.run(batch_gen)
                        print batch_mask.shape, batch_weight.shape
                        if args.weights == 1:
                            batch_mask[:, 0:1] /= batch_weight
                        elif args.weights == 2:
                            batch_mask[:, 0:1] /= np.sqrt(batch_weight)
                        if epoch == 1:
                            print 'batch_heat', np.mean(batch_heat), np.max(batch_heat), np.sum(batch_heat)
                            print 'batch_weight', batch_weight
                            print 'batch_mask', batch_mask
                            means.append(np.mean(batch_heat))
                            if i < 10:
                                print 'batch_mask', batch_mask.shape, batch_mask
                        batch_x -= mean
                        batch_x /= stddev
                        load_time += timeit.default_timer() - start
                        main_time -= timeit.default_timer()
                        myfeed = {x: batch_x, y_heat: batch_heat, phase_train: pt, learning_rate: epoch_lr,
                                  coefficients: np.array([1] + (OUT_CHANNELS-1) * [args.coeff]), mask: batch_mask}
                        res = sess.run(compute_list, feed_dict=myfeed)

                        if pt and i % 10 == 0:
                            sum = sess.run([merged], feed_dict=myfeed)
                            tensorboard_writer.add_summary(sum[0], (epoch-1) * num_batches + i)
                            tensorboard_writer.flush()
                        main_time += timeit.default_timer()
                        results.append([res[0]] + list(res[1]))
                        if i < 20000:
                            print 'train', i, res, np.mean(np.array(results, dtype=float), axis=0)
                    full_time = timeit.default_timer() - pass_start
                    rest_time = full_time - load_time - main_time
                    pass_res = np.mean(np.array(results, dtype=float), axis=0)
                    if epoch == 1:
                        print 'Heat mean', np.mean(means)
                    print 'Pass res {} time: {} main_time: {} load_time: {} rest: {}'.format(pass_res, full_time, main_time, load_time, rest_time)

                save_path = saver.save(sess, args.models_dir + '/epoch{}.ckpt'.format(epoch))
                print ('Model saved in file: %s' % save_path)

                epoch_lr *= args.epoch_decay
                epoch_time = timeit.default_timer() - epoch_start_time
                rest_time = epoch_time - train_time - load_time - valid_time
                print 'Epoch {} time {}, train {}, valid {}, load {}, dumping {}, rest {}'.\
                    format(epoch, epoch_time, train_time, valid_time, load_time, dump_time, rest_time)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ff', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--conv_stddev', type=float, default=None)
    parser.add_argument('--conv_bias', type=float, default=0.01)
    parser.add_argument('--batch_norm', type=int, default=1)
    parser.add_argument('--final_bn', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--fc_stddev', type=float, default=0.01)
    parser.add_argument('--fc_bias', type=float, default=0.01)
    parser.add_argument('--l2_reg_fc', type=float, default=0.0001)
    parser.add_argument('--epoch_decay', type=float, default=1.0)
    parser.add_argument('--predict', type=str, default=None)
    parser.add_argument('--pred_per_image', type=int, default=1000)
    parser.add_argument('--dump_threshold', type=float, default=-1.4)
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--models_dir', type=str, default='models')
    parser.add_argument('--arch', type=int, default=0)
    parser.add_argument('--dilarch', type=int, default=0)
    parser.add_argument('--arch_multiplier', type=float, default=1.)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--aug_size', type=int, default=0)
    parser.add_argument('--crop', type=int, default=256, help='Input image resolution')
    parser.add_argument('--final', type=int, default=64, help='Final image resolution')
    parser.add_argument('--augment', type=int, default=0)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--mse_mult', type=float, default=0.1)
    parser.add_argument('--nonzeros_mult', type=float, default=4)
    parser.add_argument('--band8', type=int, default=0)
    parser.add_argument('--smallval', type=int, default=0)
    parser.add_argument('--out_channels', type=int, default=2)
    parser.add_argument('--pred_iter', type=int, default=1)
    parser.add_argument('--heatmap_dir', type=str, default='heatmaps200/')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--means', type=str, default=None)
    parser.add_argument('--means_store', type=str, default=None)
    parser.add_argument('--num_dense', type=int, default=2)
    parser.add_argument('--neuron_dense', type=int, default=200)
    parser.add_argument('--filters_bound', type=int, default=200)
    parser.add_argument('--weights', type=int, default=0)
    parser.add_argument('--coeff', type=float, default=0.05)
    parser.add_argument('--valid_seed', type=int, default=123)
    parser.add_argument('--reverse_order', type=int, default=0)
    parser.add_argument('--rotate', type=int, default=0)
    parser.add_argument('--aug_colors', type=int, default=0)
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    print args
    train(args)
