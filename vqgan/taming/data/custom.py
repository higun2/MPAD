"""Paired, pre-aligned MRI volumes with explicit dataset selection."""
from pathlib import Path
from torch.utils.data import Dataset
from monai import transforms
from monai.data import Dataset as MonaiDataset

PROMPTS = {'t1': 'T1-weighted MR Image', 't2': 'T2-weighted MR Image',
           'pd': 'PD-weighted MR Image', 't1ce': 'Contrast-enhanced T1-weighted brain MRI',
           'flair': 'FLAIR brain MRI'}


def build_transform(keys, percentile=99.95, clip=True):
    return transforms.Compose([
        transforms.LoadImaged(keys=keys),
        transforms.EnsureChannelFirstd(keys=keys),
        transforms.EnsureTyped(keys=keys),
        transforms.Orientationd(keys=keys, axcodes='RAI'),
        transforms.Resized(keys=keys, spatial_size=(128, 128, 128)),
        transforms.Lambdad(keys=keys, func=lambda x: x.clamp(min=0)),
        transforms.ScaleIntensityRangePercentilesd(
            keys=keys, lower=0, upper=percentile, b_min=0, b_max=1, clip=clip),
    ])


class CustomBase(Dataset):
    def __init__(self, data_path, dataset='adni', modalities=None, param_path=None,
                 percentile=None, clip=None, **kwargs):
        if dataset not in ('adni', 'ixi', 'brats'):
            raise ValueError('dataset must be adni, ixi, or brats')
        self.modalities = modalities or (['t1', 't2', 't1ce', 'flair']
                                         if dataset == 'brats' else ['t1', 't2', 'pd'])
        root = Path(data_path)
        if not root.is_dir():
            raise FileNotFoundError(f'Dataset directory does not exist: {root}')
        records = []
        for subject in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.') and p.name != '@eaDir'):
            record = {'subject_id': subject.name}
            for modality in self.modalities:
                suffix = modality.upper() if dataset == 'ixi' else modality
                path = subject / f'{subject.name}_{suffix}.nii.gz'
                if not path.is_file():
                    raise FileNotFoundError(f'Missing {modality} volume: {path}')
                record[modality] = str(path)
            records.append(record)
        if not records:
            raise ValueError(f'No subject directories found in {root}')
        # Stage-specific defaults preserve the supplied active preprocessing.
        if percentile is None:
            percentile = 99.75 if dataset == 'adni' else 99.75
        if clip is None:
            clip = False if dataset == 'adni' else False
        self.data = MonaiDataset(records, build_transform(self.modalities, percentile, clip))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item, {**{key: PROMPTS[key] for key in self.modalities},
                      'subject_id': item['subject_id']}


class CustomTrain(CustomBase):
    pass


class CustomTest(CustomBase):
    pass
