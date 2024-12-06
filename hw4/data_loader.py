import glob
import os

import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CustomDataSet(Dataset):
    """Load images under folders"""
    def __init__(self, main_dir, ext='*.png', transform=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = glob.glob(os.path.join(main_dir, ext))
        self.total_imgs = all_imgs
        print(os.path.join(main_dir, ext))
        print(len(self))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


def get_data_loader(data_path, opts):
    """Create training and test data loaders."""
    basic_transform = transforms.Compose([
        transforms.Resize(opts.image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    advanced_transform = transforms.Compose([
        transforms.RandomCrop(opts.image_size),
        transforms.Resize(opts.image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.ColorJitter(),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if opts.data_preprocess == 'basic':
        train_transform = basic_transform
    elif opts.data_preprocess == 'advanced':
        # print("Doing advanced")
        # exit()
        train_transform = advanced_transform
        # todo: add your code here: below are some ideas for your reference
        # load_size = int(1.1 * opts.image_size)
        # osize = [load_size, load_size]
        # transforms.Resize(osize, Image.BICUBIC)
        # transforms.RandomCrop(opts.image_size)
        # transforms.RandomHorizontalFlip()
        pass

    dataset = CustomDataSet(
        os.path.join('data/', data_path), opts.ext, train_transform
    )
    dloader = DataLoader(
        dataset=dataset, batch_size=opts.batch_size,
        shuffle=True, num_workers=opts.num_workers
    )

    return dloader
