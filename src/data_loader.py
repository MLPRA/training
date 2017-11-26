import numpy as np
import json
import os

from chainer.dataset import DatasetMixin
from chainercv.utils import read_image


class DataLoader(DatasetMixin):

    def __init__(self, data_dir: 'path to dataset', label_names: 'path to label_names'):
        self.data_dir = data_dir

        with open(label_names, 'r') as f:
            self.label_names = f.read().splitlines()

        self.filepaths = []
        for root, _, files in os.walk(data_dir):
            for filename in sorted(files):
                if os.path.splitext(filename)[1] not in ['.jpg']:  # , '.png']: check chainer.utils.read_image()
                    continue

                img_filename = os.path.join(root, filename)
                anno_filename = os.path.splitext(img_filename)[0] + '.json'
                if not os.path.exists(anno_filename):
                    continue

                self.filepaths.append((img_filename, anno_filename))

    def __len__(self):
        return len(self.filepaths)

    def detect_labels(self):  # just utility
        detected_labels = []

        for img_filename, anno_filename in self.filepaths:
            with open(anno_filename) as annotation_file:
                annotations = json.load(annotation_file)
            detected_labels.append(annotations['label'])

        return list(set(detected_labels))

    def get_example(self, i):
        if i >= self.__len__():
            raise IndexError
        img = read_image(self.filepaths[i][0])
        with open(self.filepaths[i][1]) as annotation_file:
            annotations = json.load(annotation_file)  # ['labels']

        bboxes = []
        labels = []
        # for anno in annotations:
        anno = annotations  # del and indent
        x, y, h, w = (anno['boundingBox']['x'], anno['boundingBox']['y'], anno['boundingBox']['height'], anno['boundingBox']['width'])  # del ['boundingBox']
        bboxes.append([y, x, y + h, x + w])
        labels.append(self.label_names.index(anno['label']))

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return img, bboxes, labels
