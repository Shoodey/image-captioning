import numpy as np
import os, sys, getopt, time
import pickle
import caffe

caffe_root = ''

model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
layer_name = 'pool5/7x7_s1'

os.environ['GLOG_minloglevel'] = '3'

sys.path.insert(0, caffe_root + 'python')


def forward_cnn(image_path):
    image_path = image_path.strip()
    input_image = caffe.io.load_image(image_path)
    prediction = net.predict([input_image], oversample=False)
    image_vector = net.blobs[layer_name].data[0].reshape(1, -1)
    print('CNN forward pass completed')
    return image_vector


def main(argv):
    inputfile = ''
    outputfile = ''

    opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])

    for opt, arg in opts:
        if opt in ("-i"):
            inputfile = arg
        elif opt in ("-o"):
            outputfile = arg

    print('Reading images from "', inputfile)
    print('Writing vectors to "', outputfile)

    caffe.set_mode_gpu()

    net = caffe.Classifier(model_prototxt, model_trained,
                           mean=np.load(mean_path).mean(1).mean(1),
                           channel_swap=(2, 1, 0),
                           raw_scale=255,
                           image_dims=(256, 256))

    # Loading class labels
    with open(imagenet_labels) as f:
        labels = f.readlines()

    # Processing one image at a time, printing predictions and writing the vector to a file
    start = time.time()
    counter = 1
    with open(inputfile, 'r') as reader:
        for image_path, num_imgs in reader:
            print('Processing %d of %d' % (counter, num_imgs))
            if counter % 10 == 0:
                print('Time elapsed (min): %.1f' % (time.time() - start))

            image_path = image_path.strip()
            input_image = caffe.io.load_image(image_path)
            prediction = net.predict([input_image], oversample=False)
            image_vector = net.blobs[layer_name].data[0].reshape(1, -1)

            image_picklename = os.path.splitext(image_path)[0] + '.p'
            pickle.dump(image_vector, open(image_picklename, 'w'))
            counter += 1

    print('Time elapsed (s): %.4f' % (time.time() - start))
    print('Avg Time per Image (s): %.4f' % ((time.time() - start) / num_imgs))