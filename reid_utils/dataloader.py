from reid_utils.config import *
from reid_utils.augmentions import *
def get_dataset_and_dataloader(config):
    config:Config

    train_transforms = build_transforms(config, is_train=True)
    val_transforms = build_transforms(config, is_train=False)

    dataset = init_dataset(config.dataset_names,
                           root=config.root_dir)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)

    if config.sampler == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=train_collate_fn)
    else:
        train_loader = DataLoader(
            train_set, batch_size=config.batch_size,
            sampler=RandomIdentitySampler(
                dataset.train, config.batch_size, config.num_instance),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, config.num_instance),
            num_workers=config.num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size * 2, shuffle=False, num_workers=config.num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader, len(dataset.query), num_classes
