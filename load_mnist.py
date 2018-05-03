import os
import struct
import numpy as np

def load_mnist(path, mode = 'train'):
    ''' Load MNIST data from path'''
    kind = {'train':'train', 'test':'t10k'}
    labels_path = os.path.join(path,
                                "%s-labels-idx1-ubyte"% kind[mode])
    images_path = os.path.join(path,
                                "%s-images-idx3-ubyte"% kind[mode])

    with open(labels_path,'rb') as lbpath:
        # read the first 8 bytes
        magic, n = struct.unpack('>II',
                                lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        # read the first 16 bytes
        magic, num, rows, cols = struct.unpack('>IIII',
                                imgpath.read(16))
        images = np.fromfile(imgpath, dtype =np.uint8).reshape(len(labels),-1)

    return images, labels

def main():
    path = '/opt/data/pzq/MNIST'
    images, labels = load_mnist(path)
    print(images.shape)
    print(labels.shape)

if __name__ == '__main__':
    main()