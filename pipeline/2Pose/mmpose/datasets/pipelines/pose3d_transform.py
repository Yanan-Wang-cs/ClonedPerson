import mmcv
import numpy as np
import torch
from mmcv.utils import build_from_cfg

from mmpose.core.camera import CAMERAS
from mmpose.core.post_processing import fliplr_regression
from mmpose.datasets.registry import PIPELINES


@PIPELINES.register_module()
class JointRelativization:
    """Zero-center the pose around a given root joint. Optionally, the root
    joint can be removed from the origianl pose and stored as a separate item.

    Note that the relativized joints no longer align with some annotation
    information (e.g. flip_pairs, num_joints, inference_channel, etc.) due to
    the removal of the root joint.

    Args:
        item (str): The name of the pose to relativeze.
        root_index (int): Root joint index in the pose.
        remove_root (bool): If true, remove the root joint from the pose
        root_name (str): Optional. If not none, it will be used as the key to
            store the root position separated from the original pose.

    Required keys:
        item
    Modified keys:
        item, root_name
    """

    def __init__(self, item, root_index, remove_root=False, root_name=None):
        self.item = item
        self.root_index = root_index
        self.remove_root = remove_root
        self.root_name = root_name

    def __call__(self, results):
        assert self.item in results
        joints = results[self.item]
        root_idx = self.root_index

        assert joints.ndim >= 2 and joints.shape[-2] > root_idx,\
            f'Got invalid joint shape {joints.shape}'

        root = joints[..., root_idx:root_idx + 1, :]
        joints = joints - root

        if self.remove_root:
            joints = np.delete(joints, root_idx, axis=-2)
            # Add a flag to avoid latter transforms that rely on the root
            # joint or the original joint index
            results[f'{self.item}_root_removed'] = True

        results[self.item] = joints
        if self.root_name is not None:
            results[self.root_name] = root

        return results


@PIPELINES.register_module()
class JointNormalization:
    """Normalize the joint coordinate with given mean and std.

    Args:
        item (str): The name of the pose to normalize.
        mean (array): Mean values of joint coordiantes in shape [K, C].
        std (array): Std values of joint coordinates in shape [K, C].
        norm_param_file (str): Optionally load a dict containing `mean` and
            `std` from a file using `mmcv.load`.
    Required keys:
        item
    Modified keys:
        item
    """

    def __init__(self, item, mean=None, std=None, norm_param_file=None):
        self.item = item
        self.norm_param_file = norm_param_file
        if norm_param_file is not None:
            norm_param = mmcv.load(norm_param_file)
            assert 'mean' in norm_param and 'std' in norm_param
            self.mean = norm_param['mean']
            self.std = norm_param['std']
        else:
            self.mean = mean
            self.std = std

    def __call__(self, results):
        assert self.item in results
        results[self.item] = (results[self.item] - self.mean) / self.std
        return results


@PIPELINES.register_module()
class CameraProjection:
    """Apply camera projection to joint coordinates.

    Args:
        item (str): The name of the pose to apply camera projection.
        mode (str): The type of camera projection, supported options are
            - world_to_camera
            - world_to_pixel
            - camera_to_world
            - camera_to_pixel
        output_name (str|None): The name of the projected pose. If None
            (default) is given, the projected pose will be stored in place.
        camera_type (str): The camera class name (should be registered in
            CAMERA).
        camera_param (dict|None): The camera parameter dict. See the camera
            class definition for more details. If None is given, the camera
            parameter will be obtained during processing of each data sample
            with the key "camera_param".

    Required keys:
        item
        camera_param (if camera parameters are not given in initialization)
    Modified keys:
        output_name
    """

    def __init__(self,
                 item,
                 mode,
                 output_name=None,
                 camera_type='SimpleCamera',
                 camera_param=None):
        self.item = item
        self.mode = mode
        self.output_name = output_name
        self.camera_type = camera_type
        allowed_mode = {
            'world_to_camera',
            'world_to_pixel',
            'camera_to_world',
            'camera_to_pixel',
        }
        if mode not in allowed_mode:
            raise ValueError(
                f'Got invalid mode: {mode}, allowed modes are {allowed_mode}')

        if camera_param is None:
            self.static_camera = False
        else:
            self.static_camera = True
            self.camera = self._build_camera(camera_param)

    def _build_camera(self, param):
        cfgs = dict(type=self.camera_type, param=param)
        return build_from_cfg(cfgs, CAMERAS)

    def __call__(self, results):
        assert self.item in results
        joints = results[self.item]

        if self.static_camera:
            camera = self.camera
        else:
            assert 'camera_param' in results, 'Camera parameters are missing.'
            camera = self._build_camera(results['camera_param'])

        if self.mode == 'world_to_camera':
            output = camera.world_to_camera(joints)
        elif self.mode == 'world_to_pixel':
            output = camera.world_to_pixel(joints)
        elif self.mode == 'camera_to_world':
            output = camera.camera_to_world(joints)
        elif self.mode == 'camera_to_pixel':
            output = camera.camera_to_pixel(joints)
        else:
            raise NotImplementedError

        output_name = self.output_name
        if output_name is None:
            output_name = self.item

        results[output_name] = output
        return results


@PIPELINES.register_module()
class RelativeJointRandomFlip:
    """Data augmentation with random horizontal joint flip around a root joint.

    Args:
        item (str): The name of the pose to flip.
        root_index (int): Root joint index in the pose.
        visible_item (str): The name of the visibility item which will be
            flipped accordingly along with the pose.
        flip_prob (float): Probability of flip.

    Required keys:
        item
    Modified keys:
        item
    """

    def __init__(self, item, root_index, visible_item=None, flip_prob=0.5):
        self.item = item
        self.root_index = root_index
        self.vis_item = visible_item
        self.flip_prob = flip_prob

    def __call__(self, results):

        if results.get(f'{self.item}_root_removed', False):
            raise RuntimeError('The transform RelativeJointRandomFlip should '
                               f'not be applied to {self.item} whose root '
                               'joint has been removed and joint indices have '
                               'been changed')

        if np.random.rand() <= self.flip_prob:

            flip_pairs = results['ann_info']['flip_pairs']
            # flip joint coordinates
            assert self.item in results
            joints = results[self.item]

            joints_flipped = fliplr_regression(
                joints,
                flip_pairs,
                center_mode='root',
                center_index=self.root_index)

            results[self.item] = joints_flipped
            # flip joint visibility
            if self.vis_item is not None:
                assert self.vis_item in results
                visible = results[self.vis_item]
                visible_flipped = visible.copy()
                for left, right in flip_pairs:
                    visible_flipped[..., left, :] = visible[..., right, :]
                    visible_flipped[..., right, :] = visible[..., left, :]
                results[self.vis_item] = visible_flipped

        return results


@PIPELINES.register_module()
class PoseSequenceToTensor:
    """Convert pose sequence from numpy array to Tensor.

    The original pose sequence should have a shape of [T,K,C] or [K,C], where
    T is the sequence length, K and C are keypoint number and dimension. The
    converted pose sequence will have a shape of [K*C, T].

    Args:
        item (str): The name of the pose sequence

    Requred keys:
        item
    Modified keys:
        item
    """

    def __init__(self, item):
        self.item = item

    def __call__(self, results):
        assert self.item in results
        seq = results[self.item]

        assert isinstance(seq, np.ndarray)
        assert seq.ndim in {2, 3}

        if seq.ndim == 2:
            seq = seq[None, ...]

        T = seq.shape[0]
        seq = seq.transpose(1, 2, 0).reshape(-1, T)
        results[self.item] = torch.from_numpy(seq)

        return results
