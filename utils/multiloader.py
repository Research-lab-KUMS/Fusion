import pathlib
import torch
from utils.data_loader import TouchDataset
from utils.vision_dataloader import VisionDataset


class MultiDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 split='train',
                 n_frames=5,
                 do_standardization=True,
                 do_augment=False,
                 fh_v=True,
                 fh_t=False,
                 fv_t=True,
                 fv_v=False,
                 rnv=False,
                 rnt=True,
                 erase_v=False,
                 erase_t=True,
                 image_size=(200, 200)):

        self.vision_dataset = VisionDataset(data_dir,
                                            split=split,
                                            n_frames=n_frames,
                                            do_augment=do_augment,
                                            do_standardization=do_standardization,
                                            link_to_tactile=True,
                                            do_flip_h=fh_v,
                                            do_flip_v=fv_v,
                                            add_random_noise=rnv,
                                            do_erase=erase_v,
                                            image_size=image_size)

        self.tactile_dataset = TouchDataset(data_dir,
                                            split=split,
                                            n_frames=n_frames,
                                            do_augment=do_augment,
                                            do_standardization=do_standardization,
                                            do_flip_h=fh_t,
                                            do_flip_v=fv_t,
                                            add_random_noise=rnt,
                                            do_erase=erase_t)

    def __len__(self):
        return len(self.tactile_dataset)

    def __getitem__(self, index):
        tactile_data, label = self.tactile_dataset[index]
        self.vision_dataset.temp_ids = self.tactile_dataset.temp_ids
        vision_data, _ = self.vision_dataset[index]
        data_tuple = [vision_data, tactile_data]
        return data_tuple, label


if __name__ == "__main__":

    N_FRAMES = 5
    base_dir = pathlib.Path().resolve().parent
    data_dir = base_dir / 'data'

    dataset = MultiDataset(data_dir,
                           split='train',
                           n_frames=N_FRAMES,
                           do_standardization=False,
                           do_augment=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True,
                                             num_workers=0, pin_memory=False,
                                             drop_last=False)

    for d in dataloader:
        print(d)
