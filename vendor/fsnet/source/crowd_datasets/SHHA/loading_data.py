import torchvision.transforms as standard_transforms
from .SHHA import SHHA

# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def loading_data(data_root, train_patch=True, shha_fixed_side: int = 0, patch_side: int = 128):
    # data_root: str or list[str] — multiple roots merge samples from each train.list / test.list
    roots = data_root if isinstance(data_root, (list, tuple)) else [data_root]
    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])
    fs = int(shha_fixed_side) if shha_fixed_side else 0
    # create the training dataset (train_patch=False: full-image training, aligns with whole-image val)
    ps = int(patch_side) if patch_side else 128
    train_set = SHHA(
        roots,
        train=True,
        transform=transform,
        patch=train_patch,
        flip=True,
        fixed_side=fs,
        patch_side=ps,
    )
    # create the validation dataset
    val_set = SHHA(roots, train=False, transform=transform, fixed_side=fs)

    return train_set, val_set
