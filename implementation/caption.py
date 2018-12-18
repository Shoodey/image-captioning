import sys, getopt
import tensorflow as tf

from config import CaptionConfig
from model import Model
from image_feature_cnn import forward_cnn


def main(argv):
    options, arguments = getopt.getopt(argv, 'i:')

    for option, argument in options:
        if option == '-i':
            image_path = argument

    config = CaptionConfig()

    with tf.variable_scope('CNNwithLSTM') as scope:
        model = Model(config)

    saver = tf.train.Saver()

    image_vector = forward_cnn(image_path)

    with tf.Session() as session:
        save_path = 'model/model-37'
        saver.restore(session, save_path)
        print('Generating caption...')
        caption = model.generate_caption(session, image_vector)
        print('Caption: ', caption)


if __name__ == '__main__':
    main(sys.argv[1:])
