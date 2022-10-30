import random
import numpy as np
from PIL import Image
import torch
import torchvision
from utils.data_utils import prepare_metadata


class TouchDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 split='train',
                 n_frames=5,
                 specific_class=None,
                 balanced=True,
                 do_standardization=True,
                 do_augment=False,
                 do_flip_h=False,
                 do_flip_v=False,
                 flip_prob=0.5,
                 add_random_noise=False,
                 noise_mean=0.0,
                 noise_std=0.0015,
                 do_erase=False,
                 erase_prob=0.5,
                 erase_ratio=(0.5, 2),
                 erase_scale=(0.05, 0.1)):

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
                                                           value=0.0,
                                                           inplace=False)

        self.add_random_noise = add_random_noise
        self.noise_mean = noise_mean
        self.noise_std = noise_std

        metadata = prepare_metadata(data_dir, split=split, specific_class=specific_class, balanced=balanced)
        X, y, recording_ids = metadata['pressure'], metadata['objectId'], metadata['recordingId']
        self.object_ids = torch.tensor(metadata['objectId'], dtype=torch.int32)

        self.n_samples = len(y)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)
        self.recording_ids = torch.tensor(recording_ids, dtype=torch.int32)
        self.cand_dict = {}
        self.compute_sampling_dict()
        self.temp_ids = None

    def __len__(self):
        return self.n_samples

    @staticmethod
    def standardize(X):
        std, mean = torch.std_mean(X)
        X = (X - mean) / (std + 1e-7)
        return X

    @staticmethod
    def to_pillow(frame):
        return Image.fromarray(frame[0].numpy())

    @staticmethod
    def to_tensor(frame):
        return torch.tensor(np.asarray(frame).copy(), dtype=torch.float32).unsqueeze(dim=0)

    def compute_sampling_dict(self):
        for recid in torch.unique(self.recording_ids, sorted=False):
            mask = torch.eq(self.recording_ids, recid)
            candidates = self.X[mask]
            self.cand_dict[recid.item()] = candidates

    def sampler(self, frame, rec_id):
        frame = frame.unsqueeze(dim=0)
        candidates = self.cand_dict[rec_id.item()]
        random_indices = torch.randint(0, len(candidates), size=(self.n_frames - 1,))
        frames = torch.cat([frame, candidates[random_indices]], dim=0)
        self.temp_ids = random_indices
        return frames

    def __getitem__(self, index):
        frame, label, rec_id = self.X[index], self.y[index], self.recording_ids[index]
        if self.n_frames > 1:
            sampled_frames = self.sampler(frame, rec_id)
        else:
            sampled_frames = frame
        if self.do_augment is True:
            sampled_frames = self.apply_augmentation(sampled_frames)
        if self.do_standardization is True:
            sampled_frames = self.standardize(sampled_frames)
        return sampled_frames, label

    def apply_augmentation(self, frames):
        if self.do_flip_h is True or self.do_flip_v is True:
            if self.n_frames > 1:
                augmented_frames = torch.zeros_like(frames)
                prob = random.uniform(0, 1)
                for j in range(self.n_frames):
                    pil_frame = self.to_pillow(frames[j])
                    if self.do_flip_h is True:
                        if prob < self.flip_prob:
                            pil_frame = self.h_flipper(pil_frame)
                    if self.do_flip_v is True:
                        if prob < self.flip_prob:
                            pil_frame = self.v_flipper(pil_frame)
                    augmented_frames[j] = self.to_tensor(pil_frame)

            else:
                pil_frame = self.to_pillow(frames)
                prob = random.uniform(0, 1)
                if self.do_flip_h is True:
                    if prob < self.flip_prob:
                        pil_frame = self.h_flipper(pil_frame)
                if self.do_flip_v is True:
                    if prob < self.flip_prob:
                        pil_frame = self.v_flipper(pil_frame)
                augmented_frames = self.to_tensor(pil_frame)

        else:
            augmented_frames = frames

        if self.add_random_noise is True:
            mean = torch.full_like(augmented_frames, self.noise_mean)
            std = torch.full_like(augmented_frames, self.noise_std)
            augmented_frames += torch.normal(mean=mean, std=std)

        if self.do_erase is True:
            if self.n_frames > 1:
                for j in range(len(augmented_frames)):
                    augmented_frames[j] = self.eraser(augmented_frames[j])
            else:
                augmented_frames = self.eraser(augmented_frames)

        return augmented_frames


if __name__ == "__main__":
    import pathlib

    p = pathlib.Path()
    utils_dir = p.resolve()
    base_dir = utils_dir.parent
    test_dir = base_dir / 'tests' / 'dataloader test'
    test_dir.mkdir(parents=True, exist_ok=True)
    meta_path = base_dir / 'data'

    dataset = TouchDataset(meta_path,
                           split='train',
                           do_standardization=True,
                           n_frames=8,
                           specific_class=None,
                           do_augment=True,
                           do_flip_h=False,
                           do_flip_v=False,
                           flip_prob=0.5,
                           add_random_noise=False,
                           noise_mean=0.0,
                           noise_std=0.001)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
                                             num_workers=0, pin_memory=False,
                                             drop_last=False)

    for d in dataloader:
        print(d)
