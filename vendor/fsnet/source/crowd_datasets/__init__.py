# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == 'SHHA':
        from crowd_datasets.SHHA.loading_data import loading_data

        train_no_patch = getattr(args, 'train_no_patch', False)
        shha_fixed_side = int(getattr(args, "shha_fixed_side", 0) or 0)
        patch_side = int(getattr(args, 'patch_side', 128) or 128)

        def _loading_data(data_root):
            return loading_data(
                data_root,
                train_patch=not train_no_patch,
                shha_fixed_side=shha_fixed_side,
                patch_side=patch_side,
            )

        return _loading_data

    return None