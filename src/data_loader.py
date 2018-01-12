import numpy as np
import xml.etree.ElementTree as ET
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

                img_filepath = os.path.join(root, filename)
                anno_filepath = os.path.splitext(os.path.join(root, 'annotations', filename))[0] + '.xml'
                if not os.path.exists(anno_filepath):
                    continue

                self.filepaths.append((img_filepath, anno_filepath))

    def __len__(self):
        return len(self.filepaths)

    def detect_labels(self):  # just utility
        detected_labels = set({})

        for _, anno_filepath in self.filepaths:
            xml = ET.parse(anno_filepath)
            for anno in xml.findall('.object'):
                detected_labels.add(anno.find('name').text)

        return sorted(list(detected_labels))

    def get_example(self, i):
        if i >= self.__len__():
            raise IndexError
        img = read_image(self.filepaths[i][0])
        xml = ET.parse(self.filepaths[i][1])

        bboxes = []
        labels = []
        for anno in xml.findall('.object'):
            bbox = anno.find('bndbox')
            ymin = int(bbox.find('ymin').text)
            xmin = int(bbox.find('xmin').text)
            ymax = int(bbox.find('ymax').text)
            xmax = int(bbox.find('xmax').text)
            bboxes.append([ymin, xmin, ymax, xmax])
            labels.append(0)
            # labels.append(self.label_names.index(anno.find('name').text))

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return img, bboxes, labels
