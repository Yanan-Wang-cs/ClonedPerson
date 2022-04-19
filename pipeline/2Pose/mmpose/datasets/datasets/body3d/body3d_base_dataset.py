import copy
from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.data import Dataset

from mmpose.datasets.pipelines import Compose


class Body3DBaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for 3D human pose datasets.

    Subclasses should consider overwriting following methods:
        - load_config
        - load_annotations
        - build_sample_indices
        - evaluate

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.data_cfg = copy.deepcopy(data_cfg)
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.ann_info = {}

        self.load_config(self.data_cfg)
        self.data_info = self.load_annotations()
        self.sample_indices = self.build_sample_indices()
        self.pipeline = Compose(pipeline)

        self.name2id = {
            name: i
            for i, name in enumerate(self.data_info['imgnames'])
        }

    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """

        self.num_joints = data_cfg['num_joints']
        self.seq_len = data_cfg.get('seq_len', 1)
        self.seq_frame_interval = data_cfg.get('seq_frame_interval', 1)
        self.causal = data_cfg.get('causal', True)

        self.need_camera_param = False

        # create annotation information
        ann_info = {}
        ann_info['num_joints'] = self.num_joints
        ann_info['flip_pairs'] = None
        ann_info['upper_body_ids'] = None
        ann_info['lower_body_ids'] = None
        ann_info['joint_weights'] = np.full(
            self.num_joints, 1.0, dtype=np.float32)

        self.ann_info.update(ann_info)

    def load_annotations(self):
        """Load data annotation."""
        data = np.load(self.ann_file)

        # get image info
        _imgnames = data['imgname']
        num_imgs = len(_imgnames)
        num_joints = self.ann_info['num_joints']

        if 'scale' in data:
            _scales = data['scale'].astype(np.float32)
        else:
            _scales = np.zeros(num_imgs, dtype=np.float32)

        if 'center' in data:
            _centers = data['center'].astype(np.float32)
        else:
            _centers = np.zeros((num_imgs, 2), dtype=np.float32)

        # get 3D pose
        if 'S' in data.keys():
            _joints_3d = data['S'].astype(np.float32)
        else:
            _joints_3d = np.zeros((num_imgs, num_joints, 4), dtype=np.float32)

        # get 2D pose
        if 'part' in data.keys():
            _joints_2d = data['part'].astype(np.float32)
        else:
            _joints_2d = np.zeros((num_imgs, num_joints, 3), dtype=np.float32)

        data_info = {
            'imgnames': _imgnames,
            'joints_3d': _joints_3d,
            'joints_2d': _joints_2d,
            'scales': _scales,
            'centers': _centers,
        }

        return data_info

    def build_sample_indices(self):
        """Build sample indices.

        The default method creates sample indices that each sample is a single
        frame (i.e. seq_len=1). Override this method in the subclass to define
        how frames are sampled to form data samples.

        Outputs:
            sample_indices [list(tuple)]: the frame indices of each sample.
                For a sample, all frames will be treated as an input sequence,
                and the ground-truth pose of the last frame will be the target.
        """
        sample_indices = []
        if self.seq_len == 1:
            num_imgs = len(self.ann_info['imgnames'])
            sample_indices = [(idx, ) for idx in range(num_imgs)]
        else:
            raise NotImplementedError('Multi-frame data sample unsupported!')
        return sample_indices

    @abstractmethod
    def evaluate(self, outputs, res_folder, metric, **kwargs):
        """Evaluate results."""

    def prepare_data(self, idx):
        """Get data sample."""
        data = self.data_info

        frame_ids = self.sample_indices[idx]
        assert len(frame_ids) == self.seq_len

        # get the 3D/2D pose sequence
        _joints_3d = data['joints_3d'][frame_ids]
        _joints_2d = data['joints_2d'][frame_ids]

        # get the image info
        _imgnames = data['imgnames'][frame_ids]
        _centers = data['centers'][frame_ids]
        _scales = data['scales'][frame_ids]
        if _scales.ndim == 1:
            _scales = np.stack([_scales, _scales], axis=1)

        target_idx = -1 if self.causal else int(self.seq_len) // 2

        results = {
            'input_2d': _joints_2d[:, :, :2],
            'input_2d_visible': _joints_2d[:, :, -1:],
            'input_3d': _joints_3d[:, :, :3],
            'input_3d_visible': _joints_3d[:, :, -1:],
            'target': _joints_3d[target_idx, :, :3],
            'target_visible': _joints_3d[target_idx, :, -1:],
            'image_paths': _imgnames,
            'target_image_path': _imgnames[target_idx],
            'scales': _scales,
            'centers': _centers,
        }

        if self.need_camera_param:
            _cam_param = self.get_camera_param(_imgnames[0])
            results['camera_param'] = _cam_param

        return results

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.sample_indices)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        results = copy.deepcopy(self.prepare_data(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def get_camera_param(self, imgname):
        """Get camera parameters of a frame by its image name."""
        raise NotImplementedError
