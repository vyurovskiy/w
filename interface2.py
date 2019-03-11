import os
import torch

import numpy as np

from torchvision.models import inception_v3

from putils.data import normalize
from putils._mir_hook import mir
from putils.generator import TestDataset

from torch.utils.data import DataLoader

_READER = mir.MultiResolutionImageReader()
_BATCH_SIZE = os.cpu_count()
_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_WRITER = mir.MultiResolutionImageWriter()


def load_model(path_to_model):
    model = inception_v3(num_classes=1, aux_logits=False)
    model.load_state_dict(torch.load(path_to_model))
    model = model.to(_DEVICE)
    model.eval()
    return model


def get_grid(slide_path, patch_size=244):
    slide = _READER.open(slide_path)
    width, height = slide.getDimensions()
    grid = np.stack(
        np.meshgrid(
            np.arange(0, width, patch_size),
            np.arange(0, height, patch_size),
            indexing='ij'
        ),
        axis=-1
    ).reshape(-1, 2)
    return grid


if __name__ == '__main__':
    path = 'D:/ACDC_LUNG_HISTOPATHOLOGY/data/test/30.tif'
    path_to_model = 'D:/ACDC_LUNG_try2/checkpoints/19-02-22/model_inception_2277_acc=0.967.pth'
    model = load_model(path_to_model)
    grid = get_grid(path)
    testdataset = TestDataset(
        'D:/ACDC_LUNG_HISTOPATHOLOGY/data/test/30.tif', grid
    )
    testLoader = DataLoader(testdataset, batch_size=_BATCH_SIZE)
    with torch.no_grad():
        flats = sum(
            (
                (
                    model(batch.to(_DEVICE)
                         )[..., 0].sigmoid().cpu().numpy().tolist()
                ) for batch, _ in testLoader
            ), []
        )
    _WRITER.openFile('D:/ACDC_LUNG_HISTOPATHOLOGY/data/predict/30.tif')
    _WRITER.setTileSize(244)
    _WRITER.setDataType(mir.UChar)
    _WRITER.setColorType(mir.Monochrome)
    _WRITER.setCompression(mir.LZW)
    data = np.ones((244, 244), dtype='int').flatten()
    _WRITER.writeBaseImagePartToLocation(
        x=int(grid[0][0]), y=int(grid[0][1]), data=data
    )
