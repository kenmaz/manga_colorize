import cv2
import numpy as np
import sys
import os
import shutil

def all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def gen(dir, train, val, test):
    total = sum([train, val, test])
    i = 1
    it = iv = ie = 1
    for file in all_files(dir):
        i += 1
        r = i % total
        #print r,train,train+val
        if r < train:
            dst_d = "data/train"
            dst_f = "%d.jpg" % it
            it += 1
        elif train <= r and r < train+val:
            dst_d = "data/val"
            dst_f = "%d.jpg" % iv
            iv += 1
        else:
            dst_d = "data/test"
            dst_f = "%d.jpg" % ie
            ie += 1
        dst = "%s/%s" % (dst_d, dst_f)
        if not os.path.exists(dst_d):
           os.makedirs(dst_d)

        print file,dst
        shutil.move(file, dst)

if __name__ == '__main__':
    dir = sys.argv[1]
    train = int(sys.argv[2])
    val = int(sys.argv[3])
    test = int(sys.argv[4])
    gen(dir,train, val, test)
