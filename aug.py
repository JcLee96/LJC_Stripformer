import albumentations as albu

def get_transforms(size: int, scope: str = 'geometric', crop: str = 'random'):
    augs = {
        'strong': albu.Compose([
            albu.HorizontalFlip(),
            albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=20, p=.4),
            albu.ElasticTransform(),
            albu.OpticalDistortion(),
            albu.OneOf([
                albu.CLAHE(clip_limit=2),
                albu.IAASharpen(),
                albu.IAAEmboss(),
                albu.RandomBrightnessContrast(),
                albu.RandomGamma()
            ], p=0.5),
            albu.OneOf([
                albu.RGBShift(),
                albu.HueSaturationValue(),
            ], p=0.5),
        ]),
        'weak': albu.Compose([
            albu.HorizontalFlip(),
        ]),
        'geometric': albu.Compose([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.RandomRotate90(),
        ]),
        'None': None
    }

    aug_fn = augs.get(scope, None)

    crop_fn = {
        'random': albu.RandomCrop(size, size, always_apply=True),
        'center': albu.CenterCrop(size, size, always_apply=True),
        'none': None,
        None: None
    }.get(crop, None)

    # ✅ None-safe: aug/crop이 None이면 Compose에 넣지 않음
    t_list = []
    if aug_fn is not None:
        t_list.append(aug_fn)
    if crop_fn is not None:
        t_list.append(crop_fn)

    pipeline = albu.Compose(
        t_list,
        additional_targets={'target': 'image', 'sub': 'image', 'tar_sub': 'image', 'edge_sub': 'image'}
    )

    sub_resize = albu.Resize(size // 2, size // 2, always_apply=True)
    main_resize = albu.Resize(size, size, always_apply=True)

    def process(a, b, c, d, e):
        r = pipeline(image=a, target=b, sub=c, tar_sub=d, edge_sub=e)
        return (
            main_resize(image=r['image'])['image'],
            main_resize(image=r['target'])['image'],
            sub_resize(image=r['sub'])['image'],
            sub_resize(image=r['tar_sub'])['image'],
            sub_resize(image=r['edge_sub'])['image'],
        )

    return process

# ✅ dataset.py에서 텐서 변환/정규화를 처리하도록 바꿨다면 이 함수는 쓰지 않는 걸 권장
def get_normalize():
    raise RuntimeError(
        "get_normalize() is deprecated in this project. "
        "Normalization is handled inside dataset.py to keep CHW/1HW consistent."
    )