import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


# a custom dataset class for reading in our images from the list
class ReadFromList(Dataset):
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image


def create_dataloaders(batch_size, n_workers, img_dir_train, img_dir_val, img_dir_test, dataset_size, shuffle_images=True):
    # training dataset
    img_list = glob.glob(img_dir_train + '*')
    img_list.sort()
    print('Training Dataset')
    print(f'Number of images: {len(img_list)}')
    img_list = img_list[:dataset_size]

    # makes the dataset and data loader
    dataset = ReadFromList(img_list, transform=transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_images, num_workers=n_workers)
    print(f'Size of dataset: {len(dataloader_train.dataset)}')

    # validation dataset
    img_list = glob.glob(img_dir_val + '*')
    img_list.sort()
    print('Validation Dataset')
    print(f'Number of images: {len(img_list)}')
    dataset = ReadFromList(img_list, transform=transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    dataloader_val = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_images, num_workers=n_workers)
    print(f'Size of dataset: {len(dataloader_val.dataset)}')

    # test dataset
    img_list = glob.glob(img_dir_test + '*')
    img_list.sort()
    print('Testing Dataset')
    print(f'Number of images: {len(img_list)}')
    dataset = ReadFromList(img_list, transform=transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_images, num_workers=n_workers)
    print(f'Size of dataset: {len(dataloader_test.dataset)}')

    return dataloader_train, dataloader_val, dataloader_test
