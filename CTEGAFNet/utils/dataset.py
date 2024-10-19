import os
import random
import numpy as np
from PIL import Image, ImageEnhance

import torch.utils.data as data
import torchvision.transforms as transforms
import cv2

random.seed(2021)

def cv_random_flip(img, label, grad, edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        grad = grad.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, grad, edge


def randomCrop(image, label, grad, edge):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), grad.crop(random_region), edge.crop(random_region)


def randomRotation(image, label, grad, edge):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        grad = grad.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image, label, grad, edge


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, grad_root, edge_root, trainsize):
        self.trainsize = trainsize
        # get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.grads = [grad_root + f for f in os.listdir(grad_root) if f.endswith('.jpg')
                      or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(grad_root) if f.endswith('.jpg')
                     or f.endswith('.png')]

        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.grads = sorted(self.grads)
        self.edges = sorted(self.edges)

        # filter mathcing degrees of files
        self.filter_files()
        self.kernel = np.ones((5, 5), np.uint8)

        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        # get size of dataset
        self.size = len(self.images)
        print('>>> trainig/validing with {} samples'.format(self.size))

    def getFlip(self):
        p = random.randint(0, 1)
        self.flip = transforms.RandomHorizontalFlip(p)

    def __getitem__(self, index):
        self.getFlip()
        # read assest/gts/grads/depths
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        grad = self.binary_loader(self.grads[index])
        edge = self.binary_loader(self.edges[index])

        # data augumentation
        image, gt, grad, edge = cv_random_flip(image, gt, grad, edge)
        image, gt, grad, edge = randomCrop(image, gt, grad, edge)
        image, gt, grad, edge = randomRotation(image, gt, grad, edge)

        image = colorEnhance(image)
        gt = randomPeper(gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        grad = self.gt_transform(grad)
        edge = self.gt_transform(edge)
        
        return image, gt, grad, edge

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        edges = []
        gts = []
        for img_path, gt_path, edge_path in zip(self.images, self.gts, self.edges):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            edge = Image.open(edge_path)
            if img.size == gt.size and img.size == edge.size:
                images.append(img_path)
                gts.append(gt_path)
                edges.append(edge_path)
        self.images = images
        self.gts = gts
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root,gt_root, grad_root, edge_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True):
    dataset = CamObjDataset(image_root, gt_root, grad_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
