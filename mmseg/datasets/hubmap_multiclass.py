import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class HuBMAPMultiClassDataset(CustomDataset):
    """HuBMAP dataset.

    Args:
        split (str): Split txt file for HuBMAP.
    """

    CLASSES = ('background', 'kidney', 'prostate', 'largeintestine', 'spleen', 'lung')

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128]]

    def __init__(self, split, **kwargs):
        super(HuBMAPMultiClassDataset, self).__init__(
            img_suffix='.tiff', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
