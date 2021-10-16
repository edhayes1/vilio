from collections import defaultdict
import random

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from collections import Counter

from param import args

from sklearn.metrics import roc_auc_score

from fts_tsv.hm_data_tsv import load_obj_tsv

class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self, uid, sent, visual_feats=None,
                 obj_labels=None, attr_labels=None,
                 is_matched=None, label=None, vl_label=None):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.attr_labels = attr_labels
        self.is_matched = is_matched  # whether the visual and obj matched
        self.label = label


class LXMERTDataset(Dataset):
    def __init__(self, splits="train", qa_sets=None):
        super().__init__()
        self.name = splits
        self.splits = splits.split(",")

def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx),


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class LXMERTTorchDataset(Dataset):
    def __init__(self, splits="train",  topk=-1):
        super().__init__()
        self.name = splits
        self.root_dir = "/data/edward/data/"

        self.task_matched = args.task_matched

        self.samples, self.text_data = self._read_data(self.root_dir)

        print("Use %d data in torch dataset" % (len(self.samples)))
        print()
    
    def _read_data(self, directory):
        instances = []
        text_data = {}
        directory = os.path.expanduser(directory)
        for root, _, fnames in os.walk(directory, followlinks=True):
            for fname in fnames:
                if fname.lower().endswith('json'):
                    id = fname[:-5]  # remove extension to get it's unique ID
                    with open(directory + fname) as f:
                        data = json.load(f)
                    if 'src_transcript' in data:
                        data['text_data'] = [{'text': data['src_transcript']}]
                    text_data[id] = data
                    instances.append(id)

        return instances, text_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item: int):
        id = self.samples[item]
        datum = self.text_data[id]
        img_features_path = os.path.join(self.root_dir, id + '_features.npy')

        # Get image info, load first position for non-augmented
        feats = np.load(img_features_path, mmap_mode='r')[0].copy()
        obj_num = datum['num_objects'][0]
        boxes = datum['boxes'].copy()
        obj_labels = datum['object_classes'].copy()
        obj_confs = datum['object_conf'].copy()
        attr_labels = datum['attr_classes'].copy()
        attr_confs = datum['attr_conf'].copy()
        assert obj_num == len(boxes) == len(feats)

        sent = datum['text']

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = datum['size']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 1
        
        if self.task_matched:
            if random.random() < 0.5:
                is_matched = 0
                other_datum = self.data[random.randint(0, len(self.data)-1)]
                while other_datum['id'] == id:
                    other_datum = self.data[random.randint(0, len(self.data)-1)]
                sent = other_datum['text']

        label = None

        # Create target
        example = InputExample(
            id, sent, (feats, boxes),
            (obj_labels, obj_confs), (attr_labels, attr_confs),
            is_matched, label
        )
        return example