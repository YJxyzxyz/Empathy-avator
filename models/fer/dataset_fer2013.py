from torchvision import transforms, datasets

def build_transforms(img_size=224):
    # 注意：ImageFolder 默认返回 PIL.Image，因此不要再 ToPILImage()
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.1)], p=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.05,0.05), scale=(0.95,1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    return train_tf, eval_tf

def build_datasets(root="data/processed/FER2013_Images", img_size=224):
    train_tf, eval_tf = build_transforms(img_size)
    train_ds = datasets.ImageFolder(root=f"{root}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(root=f"{root}/val", transform=eval_tf)
    test_ds  = datasets.ImageFolder(root=f"{root}/test", transform=eval_tf)
    return train_ds, val_ds, test_ds
