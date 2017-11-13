#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import scipy.misc
import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import zipfile
import urllib.request
import sys
import six.moves.cPickle as pickle
from PIL import Image

from logging import getLogger, StreamHandler, INFO
logger = getLogger(__name__)
logger.setLevel(INFO)
handler = StreamHandler()
handler.setLevel(INFO)
logger.addHandler(handler)
logger.propagate=False

OKGREEN = "\033[92m"
ENDC = "\033[0m"
FAIL = "\033[91m"
WARNING="\033[93m"

class Model(object):
    def train(self, config=None):
        logger.warning("{} [WARNING] Please implement a train method {}".format(WARNING, ENDC))

    def save(self, dir, step, model_name="model"):
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.saver.save(self.sess,
                        os.path.join(dir, model_name),
                        global_step=step)
        logger.info("{} [INFO] Saved {} @ {} steps {}".format(OKGREEN, model_name, step, ENDC))
    
    def restore(self, dir):
        ckpt = tf.train.get_checkpoint_state(dir)
        
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            logger.info("Restoring {} ...".format(last_model))
            self.saver.restore(self.sess, last_model)
            #global_step = int(re.search("(\d+).", os.path.basename(last_model)).group(1))
            global_step = int(last_model.split('-')[1])
            logger.info("{} [INFO] Restored {} @ {} steps {}".format(OKGREEN, os.path.basename(last_model), global_step, ENDC))
            return True, global_step
        else:
            logger.warning("{} [WARNING] Failed to restore a model {}".format(FAIL, ENDC))
            return False, 0
        
def visualize(X, epoch, name="image"):
    #X = np.squeeze(input[0], axis=(-1,))
    #X = input[0]
    batch_size = X.shape[0]
    h = X.shape[1]
    w = X.shape[2]
    c = X.shape[3]

    height = int(np.ceil(np.sqrt(batch_size)))
    width  = int(np.ceil(np.sqrt(batch_size)))

    images = np.zeros((h*height, w*width, c))
    
    for idx, img in enumerate(X):
        i = idx % width # row
        j = idx // width # column
        images[j*h:(j+1)*h, i*w:(i+1)*w, :] = img

    images = np.squeeze(images)
    """
    if (c == 1):
        images = np.zeros((h*height, w*width))
    else:
        images = np.zeros((h*height, w*width, c))

    for idx, img in enumerate(X):
        i = idx % width # row
        j = idx // width # column

        if (c == 1):
            images[j*h:(j+1)h, i*w:(i+1)*w] = img
        else:
            images[j*h:(j+1)h, i*w:(i+1)*w, :] = img
    """     
    images = ((images*127.5) + 127.5).astype(np.uint8)
    images = cv2.resize(images, dsize=(128*width, 128*height))
    cv2.imwrite('{}_{}.png'.format(name, epoch), images)
    return images

def save_gif(inputs, filename="image"):
    img = Image.fromarray(inputs[0])
    imgs = [Image.fromarray(x) for x in inputs[1:]]
    img.save("{}.gif".format(filename), format="GIF",
             save_all=True, append_images=imgs, loop=1000, duration=500)

def plot(data, index, title='data', x_label='X', y_label='Y'):
    #font = {'family' : 'Helvetica'}
    #matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=2)
    plt.style.use('ggplot')
    df = pd.DataFrame(data, index=index)
    df.plot(title=title, fontsize=13, alpha=0.75, rot=45, figsize=(15, 10))
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)

    plt.savefig("{}.jpg".format(title))


# ====================
# Data Loader 
# ====================
DATA_URL="http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data/animeface-character-dataset.zip"
""" from https://github.com/nagadomi/lbpcascade_animeface """
CASCADE_PATH = './lbpcascade_animeface.xml'

class Loader:
    def __init__(self, data_dir, data_name, batch_size):
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.cur = 0
        
        self._load_data(data_dir, data_name)
        self.batch_num = int(len(self.data) / batch_size)
        self.data_num = len(self.data)        

    def reset(self):
        np.random.shuffle(self.data)
        self.cur = 0
        
    def next_batch(self):
        if self.cur + self.batch_size > self.data_num:
            #return None
            self.cur = 0

        batch = self.data[self.cur:self.cur+self.batch_size]
        self.cur += self.batch_size
        return batch

    def _download_data(self, filename, filepath, data_dir):
    
        """ Download data and unzip """
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %(filename,
                                                            float(count * block_size) / float(total_size) *100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        statinfo = os.stat(filepath)

        logger.info("{} [INFO] Data downloaded {} {} bytes {}".format(OKGREEN, filename, statinfo.st_size, ENDC))
                    
        with zipfile.ZipFile(filepath, 'r') as zip_:
            logger.info(" [INFO] Unzipping ...")
            zip_.extractall(path=data_dir)
            os.remove(filepath)

    def _load_data(self, data_dir, data_name):
    
        data_dir = os.path.join(data_dir, data_name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            self._download_data(filename, filepath, data_dir)

        pkl_path = os.path.join(data_dir, "data.pkl")

        if os.path.exists(pkl_path):
            logger.info(" [INFO] Loading from pkl...")
            self.data = pickle.load(open(pkl_path, "rb"))
            self.data_num = len(self.data)
        else:
            self.data_num = 0
            data_dir = os.path.join(data_dir, filename.split('.')[0], "thumb")
            dirs = os.listdir(data_dir)
            data = []
            for i, d in enumerate(dirs):
                files = os.listdir(os.path.join(data_dir, d))
        
                sys.stdout.write("\rDirectories: {}/{}".format(i, len(dirs)))
                sys.stdout.flush()
        
                for f in files:
                    root, ext = os.path.splitext(f)
                    if ext == ".png":
                        # BGR
                        img = cv2.imread(os.path.join(data_dir, d, f))
                        # BGR2RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        gray = cv2.equalizeHist(gray)
                        cascade = cv2.CascadeClassifier(CASCADE_PATH)
                        face = cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=2,
                            minSize=(10, 10))
                
                        #print("{}".format(face))
                        if len(face) == 1:
                            x, y, w, h = face[0]
                            img = img[y:y+h, x:x+w]
                            img = cv2.resize(img, (112, 112))
                            
                            data.append(img)
                            #data.append(img.transpose(2, 0, 1))
                            #print("{}".format(img.shape))
                            #cv2.namedWindow('window')
                            #cv2.imshow('window', img)
                            #cv2.waitKey(0)
                            #cv2.destroyAllWindows()
                            self.data_num += 1

            self.data = np.asarray(data, dtype=np.float32)
            pickle.dump(data, open(pkl_path, "wb"), -1)
