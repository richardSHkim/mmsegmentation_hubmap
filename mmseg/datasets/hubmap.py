import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class HuBMAPDataset(CustomDataset):
    """HuBMAP dataset.

    Args:
        split (str): Split txt file for HuBMAP.
    """

    CLASSES = ('background', 'tissue')

    PALETTE = [[0, 0, 0], [128, 0, 0]]

    def __init__(self, split, **kwargs):
        super(HuBMAPDataset, self).__init__(
            img_suffix='.tiff', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
