from typing import Dict, Any, Tuple
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np

class LesionDataset(Dataset):
    def __init__(self, csv_path: str, image_key: str, mask_key: str, size=(128,128)):
        self.df = pd.read_csv(csv_path)
        self.image_key = image_key
        self.mask_key = mask_key
        self.size = size
    def __len__(self):
        return len(self.df)
    def _load_img(self, path: str) -> np.ndarray:
        im = Image.open(path).convert("RGB").resize(self.size, resample=Image.BILINEAR)
        return np.asarray(im)
    def _load_mask(self, path: str) -> np.ndarray:
        mk = Image.open(path).convert("L").resize(self.size, resample=Image.NEAREST)
        mk = (np.asarray(mk) > 127).astype(np.float32)
        return mk
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = self._load_img(row[self.image_key]).copy()
        y = self._load_mask(row[self.mask_key]).copy()
        x = torch.from_numpy(x).permute(2,0,1).float() / 255.0
        y = torch.from_numpy(y)[None, ...].float()
        return {"image": x, "mask": y}
    
    def _load_mask(self, path: str) -> np.ndarray:
        mk = Image.open(path).convert("L").resize(self.size, resample=Image.NEAREST)
        mk = (np.asarray(mk) > 127).astype(np.float32)
        return mk
