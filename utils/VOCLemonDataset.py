import torch
import torchvision
from pathlib import Path
import cv2
from utils.xmlparser import get_bboxes, get_all_classes
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms.functional import resize, to_tensor
class VOCLemon(Dataset):
    def __init__(self, 
                 image_list_path : str | Path, 
                 image_path : str | Path, 
                 annot_path : str | Path):
        with open(image_list_path) as f:
            self.image_list = f.read().splitlines()
        self.image_path = Path(image_path)
        self.annot_path = Path(annot_path)
        self.label_encoder = LabelEncoder()
        self.labels_str = set()

        for image_annot in self.image_list:
            classes = get_all_classes(self.annot_path / (image_annot + ".xml"))
            self.labels_str = self.labels_str.union(classes)

        self.label_encoder.fit(list(self.labels_str))

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        xml_path = self.annot_path / (image_name + ".xml")
        img_name, names, bboxes = get_bboxes(xml_path)
        names = self.label_encoder.transform(names)
        target = dict()
        target['boxes'] = bboxes
        target['labels'] = names

        img = Image.open(self.image_path / (img_name)).convert('RGB')
        img = to_tensor(img)
        #img = torchvision.transforms.functional.pil_to_tensor(img)
        old_w, old_h = img.shape[1], img.shape[2]
        img = resize(img, size=1000)
        new_w, new_h = img.shape[1], img.shape[2]
        n_bboxes = target['boxes'].shape[0]
        for idx in range(n_bboxes):
            target['boxes'][idx, 0::2] *= new_w/old_w
            target['boxes'][idx, 1::2] *= new_h/old_h
        return img, target
    
    def get_label_from_encoding(self, cls_id : int) -> str:
        return self.label_encoder.inverse_transform([cls_id])[0]