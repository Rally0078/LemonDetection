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
    """
        PyTorch Dataset to read the Lemon dataset in Pascal VOC format.
        Probably could use it for any Pascal VOC format images with some work.
    """
    def __init__(self, 
                 image_list_path : str | Path, 
                 image_path : str | Path, 
                 annot_path : str | Path):
        """
        Constructor for the PyTorch Dataset to read VOC format.
        ### Parameters
            image_list_path : str | Path
                    -   string or Path object to a text file containing a list of image file names.

            image_path : str | Path
                    -   string or Path object to the directory containing the listed images.

            annot_path : str | Path
                    -   string or Path object to the directory containing the annotations corresponding to the images
        """

        # Get list of images
        with open(image_list_path) as f:
            self.image_list = f.read().splitlines()
        self.image_path = Path(image_path)
        self.annot_path = Path(annot_path)
        self.label_encoder = LabelEncoder()

        # Get all unique labels and encode them into integers
        self.labels_str = set()
        for image_annot in self.image_list:
            classes = get_all_classes(self.annot_path / (image_annot + ".xml"))
            self.labels_str = self.labels_str.union(classes)

        # Fit the label encoder with the class names
        self.label_encoder.fit(list(self.labels_str))

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):

        # Get image name for the given idx
        image_name = self.image_list[idx]

        # Get bounding box from corresponding xml file
        xml_path = self.annot_path / (image_name + ".xml")
        img_name, cls_names, bboxes = get_bboxes(xml_path)
        cls_encoded_names = self.label_encoder.transform(cls_names)
        target = dict()
        target['boxes'] = bboxes
        target['labels'] = cls_encoded_names

        # Resize image and bounding boxes to standard Faster RCNN size
        # PyTorch format (C, H, W)
        # PIL Image format (W, H)
        # boxes format (W_min, H_min, W_max, H_max)
        img = Image.open(self.image_path / (img_name)).convert('RGB')   #PIL Image format (W,H)
        img = to_tensor(img)    #PyTorch format (C,H,W)
        old_h, old_w = img.shape[1], img.shape[2]
        img = resize(img, size=1000)
        new_h, new_w = img.shape[1], img.shape[2]
        n_bboxes = target['boxes'].shape[0]
        for idx in range(n_bboxes):
            target['boxes'][idx, 0::2] *= new_w/old_w
            target['boxes'][idx, 1::2] *= new_h/old_h
        
        return img, target
    
    def get_label_from_encoding(self, cls_id : int) -> str:
        """
        Returns label string from the integer encoded class label.
        ### Parameters
            cls_id : int
                    - Integet encoded class label.
        ### Returns
            A string containing the class name corresponding to the encoded label.
        """
        return self.label_encoder.inverse_transform([cls_id])[0]