import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple
import torch

def get_bboxes(xml_path : str | Path ) -> Tuple[str, list[str], torch.FloatTensor]:
    """
    ### Parameters
        xml_path : str | Path 
                - string or Path object to the XML file
    ### Returns

        str - string containing image name

        list[str] - List of strings containing class names

        a PyTorch Tensor containing all the bounding boxes in x1, y1, x2, y2 format (4 corners)
    """
    parsed_xml = ET.parse(xml_path)
    root = parsed_xml.getroot()
    b = root.findall('object')
    img_name = root.find('filename').text
    bboxes = torch.from_numpy(np.empty(shape=(len(b), 4)))
    classes = []
    for i, boxes in enumerate(root.iter('object')):
        bboxes[i,0] = float(boxes.find('bndbox/xmin').text)
        bboxes[i,1] = float(boxes.find('bndbox/ymin').text)

        bboxes[i,2] = float(boxes.find('bndbox/xmax').text)
        bboxes[i,3] = float(boxes.find('bndbox/ymax').text)
        classes.append(boxes.find('name').text)
    return img_name, classes, bboxes

def get_all_classes(xml_path : str | Path) -> set[str]:
    """
    ### Parameters:
    xml_path : str | Path
            -   string or Path object to the XML file
    ### Returns:
        - a set containing all the unique names of classes in the XML file
    """
    unique_classes = set()
    parsed_xml = ET.parse(xml_path)
    root = parsed_xml.getroot()
    for boxes in root.iter('object'):
        unique_classes.add(boxes.find('name').text)
    return unique_classes