# -*- coding: utf-8 -*-
"""
detectors/arcface_model.py
--------------------------
ArcFace model architecture from https://github.com/shamma315/dronefacerecognition

Contains the Embeddinghead projection network used to refine InceptionResnetV1
embeddings from 512-d to 256-d for face recognition.

ArcFaceLoss is included for completeness but is only needed during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embeddinghead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)           # keep only positive values
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.normalize(x, p=2, dim=1)  # L2-normalise: all embeddings same length
        return x


class ArcFaceLoss(nn.Module):
    """ArcFace margin loss — used during training only, not needed for inference."""

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.s = 64.0   # scaling factor
        self.m = 0.5    # angular margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)  # one learnable vector per person

    def forward(self, embeddings, labels):
        cosine_theta = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(cosine_theta.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot_encode = torch.zeros_like(cosine_theta)
        one_hot_encode.scatter_(1, labels.view(-1, 1), 1)
        output = torch.where(one_hot_encode.bool(), torch.cos(theta + self.m), cosine_theta)
        return output * self.s
