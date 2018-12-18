from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import os
import json
import random
import pickle
import numpy as np


def generate_captions_val(session, model, epoch, valSize=200, debug=False):
    with open('data/val_image_id2feature.pkl', 'r') as f:
        val_image_id2feature = pickle.load(f)

    total_images = len(val_image_id2feature)
    results_list = []
    val_set = val_image_id2feature.items()
    for step, (img_id, img_feature) in enumerate(val_set[:valSize]):
        generated_caption = model.generate_caption(session, img_feature)
        line = {}
        line['image_id'] = img_id
        line['caption'] = generated_caption
        results_list.append(line)

        print(        '%d/%d imgid %d: %s' % (step, valSize, img_id, generated_caption)

    results_dir = '%s/results' % model.config.model_name
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    fn = '%s/val_res_%d.json' % (results_dir, epoch)
    with open(fn, 'w') as f:
        json.dump(results_list, f, sort_keys=True, indent=4)
    print(    'json results dumped in %s' % fn


def evaluateModel(model_json):
    cocoRes = coco.loadRes(model_json)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()  # to evaluate only on subset of images
    cocoEval.evaluate()
    results = {}
    for metric, score in cocoEval.eval.items():
        results[metric] = score
    del cocoRes
    del cocoEval
    return results


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def get_batches(train_captions, batch_size):
    train_batches = []
    for sent_length, caption_set in train_captions.items():
        caption_set = list(caption_set)
        random.shuffle(caption_set)
        num_captions = len(caption_set)
        num_batches = num_captions // batch_size
        for i in range(num_batches + 1):
            end_idx = min((i + 1) * batch_size, num_captions)
            new_batch = caption_set[(i * batch_size):end_idx]
            if len(new_batch) == batch_size:
                train_batches.append((new_batch, sent_length))
    random.shuffle(train_batches)
    return train_batches


def formatPlaceholder(batch_item, batch_size, img_dim, train_caption_id2sentence, train_caption_id2image_id,
                      train_image_id2feature):
    (caption_ids, sent_length) = batch_item
    num_captions = len(caption_ids)
    sentences = np.array([train_caption_id2sentence[k] for k in caption_ids])
    images = np.array([train_image_id2feature[train_caption_id2image_id[k]] for k in caption_ids])
    targets = sentences[:, 1:]

    sentences_template = np.zeros([batch_size, sent_length])
    images_template = np.zeros([batch_size, img_dim])
    targets_template = -np.ones([batch_size, sent_length + 1])

    sentences_template[range(num_captions), :] = sentences
    images_template[range(num_captions), :] = images
    targets_template[range(num_captions), 1:sent_length] = targets
    assert (targets_template[:, [0, -1]] == -1).all()  # front and back should be padded with -1

    return (sentences_template, images_template, targets_template)


def train_data_iterator(train_captions, train_caption_id2sentence, train_caption_id2image_id, train_image_id2feature,
                        config):
    batch_size = config.batch_size
    img_dim = config.img_dim

    train_batches = get_batches(train_captions, batch_size)
    for batch_item in train_batches:
        sentences, images, targets = formatPlaceholder(batch_item, batch_size, img_dim, train_caption_id2sentence,
                                                       train_caption_id2image_id, train_image_id2feature, )
        yield (sentences, images, targets)