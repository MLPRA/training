import os
import json
import lxml.etree as ET
from PIL import Image


def convert_json_to_xml(imagefolder, jsonfolder, xml_dir=None):
    jpg_paths, json_paths = {}, {}
    for dir_path in set([imagefolder, jsonfolder]):
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                if file_extension.lower() == '.jpg':
                    jpg_paths[file_name] = '{}/{}'.format(root, file)
                elif file_extension.lower() == '.json':
                    json_paths[file_name] = '{}/{}'.format(root, file)

    for name, json_path in json_paths.items():
        if name not in jpg_paths:
            continue
        jpg_path = jpg_paths[name]

        with open(json_paths[name]) as file:
            data = json.load(file)
        with Image.open(jpg_paths[name]) as img:
            width, height = img.size

        label = data['label']
        xmin = int(data['boundingBox']['x'] * width)
        ymin = int(data['boundingBox']['y'] * height)
        xmax = int(data['boundingBox']['x'] + data['boundingBox']['width'] * width)
        ymax = int(data['boundingBox']['y'] + data['boundingBox']['height'] * height)

        e_root = ET.Element('annotation')

        e_size = ET.SubElement(e_root, 'size')
        e_width = ET.SubElement(e_size, 'width')
        e_width.text = str(width)
        e_height = ET.SubElement(e_size, 'height')
        e_height.text = str(height)

        e_object = ET.SubElement(e_root, 'object')

        e_label = ET.SubElement(e_object, 'name')
        e_label.text = label

        e_bndbox = ET.SubElement(e_object, 'bndbox')
        e_xmin = ET.SubElement(e_bndbox, 'xmin')
        e_xmin.text = str(xmin)
        e_ymin = ET.SubElement(e_bndbox, 'ymin')
        e_ymin.text = str(ymin)
        e_xmax = ET.SubElement(e_bndbox, 'xmax')
        e_xmax.text = str(xmax)
        e_ymax = ET.SubElement(e_bndbox, 'ymax')
        e_ymax.text = str(ymax)

        if xml_dir is None:
            xml_dir = os.path.dirname(json_path) + '/xml/'
            if not os.path.exists(xml_dir):
                os.makedirs(xml_dir)
        xml_path = xml_dir + name + '.xml'
        ET.ElementTree(e_root).write(xml_path, pretty_print=True)
