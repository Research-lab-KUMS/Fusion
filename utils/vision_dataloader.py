import pathlib
import random
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision
from skimage import io as skio
from utils.data_utils import prepare_metadata


class VisionDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 split='train',
                 n_frames=5,
                 do_standardization=True,
                 link_to_tactile=False,
                 do_augment=False,
                 do_flip_h=False,
                 do_flip_v=False,
                 flip_prob=0.5,
                 add_random_noise=False,
                 noise_mean=0,
                 noise_std=15,
                 do_erase=False,
                 erase_prob=0.5,
                 erase_ratio=(0.5, 2),
                 erase_scale=(0.05, 0.1),
                 image_size=(200, 200)):

        self.metadata = prepare_metadata(data_dir, split=split, balanced=True)
        self.data_dir = data_dir
        self.image_size = image_size
        self.do_standardization = do_standardization
        self.do_augment = do_augment
        self.split = split
        self.n_frames = n_frames
        self.do_flip_h = do_flip_h
        self.do_flip_v = do_flip_v
        self.flip_prob = flip_prob
        self.h_flipper = torchvision.transforms.functional.hflip
        self.v_flipper = torchvision.transforms.functional.vflip
        self.do_erase = do_erase
        self.erase_prob = erase_prob
        self.erase_ratio = erase_ratio
        self.erase_scale = erase_scale
        self.eraser = torchvision.transforms.RandomErasing(p=erase_prob,
                                                           scale=erase_scale,
                                                           ratio=erase_ratio,
                                                           value=(0, 0, 0),
                                                           inplace=False)

        self.add_random_noise = add_random_noise
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.cand_dict = {}
        self.compute_sampling_dict()
        self.link_to_tactile = link_to_tactile
        self.temp_ids = None

    def __len__(self):
        return len(self.metadata['objectId'])

    def standardize(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X.copy()).float()
        if isinstance(X, Image.Image):
            X = torch.tensor(np.asarray(X.copy())).float()
        if X.dtype != torch.float32:
            X = X.float()
        if self.n_frames > 1:
            X = X.permute(0, 3, 1, 2).contiguous()
        else:
            X = X.permute(2, 0, 1).contiguous()

        std, mean = torch.std_mean(X)
        return (X - mean) / (std + 1e-7)

    @staticmethod
    def to_pillow(frame):
        return Image.fromarray(frame)

    @staticmethod
    def to_tensor(frame, dtype=None):
        if dtype is None:
            tensor_ = torch.tensor(np.asarray(frame).copy())
        else:
            tensor_ = torch.tensor(np.asarray(frame).copy(), dtype=dtype)
        return tensor_

    def resize(self, frame):
        resized_frame = np.asarray(self.to_pillow(frame).resize(self.image_size))
        return resized_frame

    def image_loader(self, path, backend='pil'):
        path = str(path)
        if backend == 'cv':
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        elif backend == 'skio':
            image = skio.imread(path)
        elif backend == 'pil':
            image = np.asarray(Image.open(path))
        else:
            raise ValueError('"backend" Argument must be either opencv, skio or pil')
        image = self.resize(image)
        return image

    def compute_sampling_dict(self):
        recording_ids = self.metadata['recordingId']
        for recid in np.unique(recording_ids):
            candidates_masks = (recording_ids == recid)
            self.cand_dict[recid] = candidates_masks

    def sampler(self, index):
        recid = self.metadata['recordingId'][index]
        recordings = self.metadata['recordings']
        label = torch.tensor(self.metadata['objectId'][index], dtype=torch.int64)
        batch_id = self.metadata['batchId'][index]
        batch_names = self.metadata['batches']
        offset_frame_number = self.metadata['frame'][index]

        offset_path = self.data_dir / batch_names[batch_id].item() / recordings[recid].item()
        offset_frame_path = offset_path / '{:0>6}.jpg'.format(offset_frame_number)
        offset_frame = self.image_loader(offset_frame_path)

        if self.n_frames > 1:
            candidates_mask = self.cand_dict[recid]
            if self.link_to_tactile is True:
                random_indices = self.temp_ids
            else:
                random_indices = np.random.randint(0, sum(candidates_mask), size=(self.n_frames-1, ), dtype=np.int32)

            frames = [offset_frame]
            for k in range(self.n_frames - 1):
                ind = random_indices[k]
                frame_number = self.metadata['frame'][candidates_mask][ind]
                frame_path = offset_path / '{:0>6}.jpg'.format(frame_number)
                frame = self.image_loader(frame_path)
                frames.append(frame)
            frames = np.stack(frames, axis=0)
        else:
            frames = offset_frame

        return frames, label

    def __getitem__(self, index):
        sampled_frames, label = self.sampler(index)
        # print(sampled_frames.max(), sampled_frames.min(), sampled_frames.mean())
        if self.do_augment is True:
            sampled_frames = self.apply_augmentation(sampled_frames)
        if self.do_standardization is True:
            sampled_frames = self.standardize(sampled_frames)
        return sampled_frames, label

    def apply_augmentation(self, frames):
        if self.do_flip_h is True or self.do_flip_v is True:
            if self.n_frames > 1:
                augmented_frames = torch.zeros(frames.shape, dtype=torch.float32)
                for i in range(self.n_frames):
                    pil_frame = self.to_pillow(frames[i])
                    if self.do_flip_h is True:
                        prob = random.uniform(0, 1)
                        if prob < self.flip_prob:
                            pil_frame = self.h_flipper(pil_frame)
                    if self.do_flip_v is True:
                        prob = random.uniform(0, 1)
                        if prob < self.flip_prob:
                            pil_frame = self.v_flipper(pil_frame)
                    augmented_frames[i] = self.to_tensor(pil_frame)

            else:
                pil_frame = self.to_pillow(frames)
                if self.do_flip_h is True:
                    prob = random.uniform(0, 1)
                    if prob < self.flip_prob:
                        pil_frame = self.h_flipper(pil_frame)
                if self.do_flip_v is True:
                    prob = random.uniform(0, 1)
                    if prob < self.flip_prob:
                        pil_frame = self.v_flipper(pil_frame)
                augmented_frames = self.to_tensor(pil_frame, dtype=torch.float32)

        else:
            augmented_frames = self.to_tensor(frames, dtype=torch.float32)

        if self.add_random_noise is True:
            mean = torch.full_like(augmented_frames, self.noise_mean)
            std = torch.full_like(augmented_frames, self.noise_std)
            augmented_frames += torch.normal(mean=mean, std=std)
            augmented_frames = torch.clip(augmented_frames, min=0, max=255).to(torch.uint8)

        if self.do_erase is True:
            if self.n_frames > 1:
                for j in range(len(augmented_frames)):
                    f = augmented_frames[j].permute(2, 0, 1).contiguous()
                    f = self.eraser(f)
                    augmented_frames[j] = f.permute(1, 2, 0).contiguous()
            else:
                f = augmented_frames.permute(2, 0, 1).contiguous()
                f = self.eraser(f)
                augmented_frames = f.permute(1, 2, 0).contiguous()

        return augmented_frames


if __name__ == "__main__":

    base_dir = pathlib.Path().resolve().parent
    data_dir = base_dir / 'data'

    dataset = VisionDataset(data_dir,
                            split='train',
                            do_standardization=False,
                            n_frames=8,
                            do_augment=True,
                            do_flip_h=True,
                            do_flip_v=False,
                            flip_prob=0.5,
                            add_random_noise=True,
                            do_erase=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
                                             num_workers=0, pin_memory=False,
                                             drop_last=False)

    for d in dataloader:
        print(d)
