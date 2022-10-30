import numpy as np
from scipy.io import loadmat


def prepare_metadata(data_dir, split='train', specific_class=None, balanced=True):

    metapath = data_dir / 'metadata.mat'
    metadata = loadmat(metapath)

    metadata['pressure'] = metadata['pressure'][:, np.newaxis].astype(np.float32)
    metadata['pressure'] = np.clip((metadata['pressure'] - 500) / (650 - 500), 0.0, 1.0)

    metadata['objectId'] = metadata['objectId'].flatten().astype(np.uint8)
    metadata['isBalanced'] = metadata['isBalanced'].flatten()
    metadata['recordingId'] = metadata['recordingId'].flatten().astype(np.int64)
    metadata['splitId'] = metadata['splitId'].flatten()
    metadata['hasValidLabel'] = metadata['hasValidLabel'].flatten()
    mask = metadata['hasValidLabel'].copy()

    if split == 'test':
        mask = np.logical_and(mask, metadata['splitId'].flatten())
    elif split == 'train':
        mask = np.logical_and(mask, np.logical_not(metadata['splitId'].flatten()))
    else:
        raise ValueError('split argument must be train or test, {} was given'.format(split))

    if balanced is True:
        mask = np.logical_and(mask, metadata['isBalanced'])

    if specific_class is not None:
        class_mask = metadata['objectId'] == specific_class
        mask = np.logical_and(mask, class_mask)

    indices = np.argwhere(mask).flatten()

    for k in metadata.keys():
        if k.startswith('__'):
            continue
        elif k in ['batches', 'splits', 'recordings', 'objects', 'actionStartTS', 'actionEndTS']:
            metadata[k] = metadata[k].flatten()
        else:
            metadata[k] = metadata[k][indices]
            if k != 'pressure':
                metadata[k] = metadata[k].flatten()

    return metadata


if __name__ == '__main__':

    import pathlib

    utils_dir = pathlib.Path().resolve()
    base_dir = utils_dir.parent
    data_dir = base_dir / 'data'
    meta_train = prepare_metadata(data_dir, split='train', balanced=True, specific_class=None)
    meta_test = prepare_metadata(data_dir, split='test', balanced=True, specific_class=None)
