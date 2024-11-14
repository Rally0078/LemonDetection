import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

def get_bboxes(xml_path : str | Path ) -> Tuple[str, list[str], np.ndarray]:
    """
    Input: xml_path : string or Path object to the XML file
    Output: a Numpy array containing all the bounding boxes in x1,y1, x2,y2 format(4 corners)
    """
    parsed_xml = ET.parse(xml_path)
    root = parsed_xml.getroot()
    b = root.findall('object')
    img_name = root.find('filename').text
    bboxes = np.empty(shape=(len(b), 4))
    classes = []
    for i, boxes in enumerate(root.iter('object')):
        bboxes[i,0] = boxes.find('bndbox/xmin').text
        bboxes[i,1] = boxes.find('bndbox/ymin').text

        bboxes[i,2] = boxes.find('bndbox/xmin').text
        bboxes[i,3] = boxes.find('bndbox/ymin').text
        classes.append(boxes.find('name').text)
    return img_name, classes, bboxes

def get_all_classes(xml_path : str | Path) -> set[str]:
    unique_classes = set()
    parsed_xml = ET.parse(xml_path)
    root = parsed_xml.getroot()
    b = root.findall('object')
    for boxes in root.iter('object'):
        unique_classes.add(boxes.find('name').text)
    return unique_classes