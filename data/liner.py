import cv2
import numpy as np
import sys
import os

def make_contor_image(src, dst, w = 5):
    print "%s, %s" % (src, dst)
    neiborhood24 = np.array(np.ones([w,w]), np.uint8)
    gray = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    dilated = cv2.dilate(gray, neiborhood24, iterations=1)
    diff = cv2.absdiff(dilated, gray)
    contour = 255 - diff

    out_img = cv2.cvtColor(contour, cv2.COLOR_GRAY2RGB)
    org_img = cv2.imread(src)
    concat = cv2.hconcat([org_img, out_img])
    cv2.imwrite(dst, concat)

def all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def main(root):
    dst_dir = "imgs_out"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    i = 0
    for file in all_files(root):
        i += 1
        root, ext = os.path.splitext(file)
        if ext == ".jpg":
            dst = "%s/%d%s" % (dst_dir, i, ext)
            make_contor_image(file, dst)

if __name__ == '__main__':
    main(sys.argv[1])
    #make_contor_image(sys.argv[1], "out.jpg")

