import os
import torch
import argparse
import numpy as np
from scipy import misc
import cv2
import imageio
from PIL import Image

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils.dataset import test_dataset as EvalDataset
from CTEGAFNet import CTEGAFNet as Network

def evaluator(model, val_root, map_save_path, trainsize=384):
    val_loader = EvalDataset(image_root=val_root + 'Imgs/',
                             gt_root=val_root + 'GT/',
                             testsize=trainsize)

    model.eval()
    with torch.no_grad():
        for i in range(val_loader.size):
            image, gt, name, _ = val_loader.load_data()
            gt = np.asarray(gt, np.float32)

            image = image.cuda()

            output = model(image)
            output = F.upsample(output[0], size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
            output = Image.fromarray((output * 255).astype(np.uint8))
            imageio.imsave(map_save_path + name, output)
            # misc.imsave(map_save_path + name, output)
            print('>>> saving prediction at: {}'.format(map_save_path + name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='convnextv2_base', 
                        choices=['DGNet', 'DGNet-S', 'DGNet-PVTv2-B0', 'DGNet-PVTv2-B1', 'DGNet-PVTv2-B2', 'DGNet-PVTv2-B3', 'DGNet-PVTv2-B4','convnextv2_base','convnextv2_tiny','unireplknet_s'])
    parser.add_argument('--snap_path', type=str, default='./snapshot/CTEGAFNet_MICAI/Net_epoch_best.pth',
                        help='train use gpu')	#模型保存的路径
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='train use gpu')
    opt = parser.parse_args()

    txt_save_path = './result/{}/'.format(opt.snap_path.split('/')[-2])
    os.makedirs(txt_save_path, exist_ok=True)

    print('>>> configs:', opt)

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    cudnn.benchmark = True
    if opt.model == 'DGNet':
        model = Network(channel=64, arc='EfficientNet-B4', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'DGNet-S':
        model = Network(channel=32, arc='EfficientNet-B1', M=[8, 8, 8], N=[8, 16, 32]).cuda()
    elif opt.model == 'DGNet-PVTv2-B0':
        model = Network(channel=32, arc='PVTv2-B0', M=[8, 8, 8], N=[8, 16, 32]).cuda()
    elif opt.model == 'DGNet-PVTv2-B1':
        model = Network(channel=64, arc='PVTv2-B1', M=[8, 8, 8], N=[4, 8, 16]).cuda()   
    elif opt.model == 'DGNet-PVTv2-B2':
        model = Network(channel=64, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'DGNet-PVTv2-B3':
        model = Network(channel=64, arc='PVTv2-B3', M=[8, 8, 8], N=[4, 8, 16]).cuda()   
    elif opt.model == 'DGNet-PVTv2-B4':
        model = Network(channel=64, arc='PVTv2-B4', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'convnextv2_base':
        print('--> using convnextv2_base right now')
        model = Network(arc='convnextv2_base').cuda()
    elif opt.model == 'convnextv2_tiny':
        print('--> using Convnext right now')
        model = Network(arc='convnextv2_tiny').cuda()
    elif opt.model == 'unireplknet_s':
        print('--> using unireplknet_s right now')
        model = Network(arc='unireplknet_s').cuda()
    else:
        raise Exception("Invalid Model Symbol: {}".format(opt.model))
    
    # TODO: remove FC layers from snapshots
    model.load_state_dict(torch.load(opt.snap_path), strict=False)
    model.eval()
 # 'COD10K', 'NC4K'
    # for data_name in ['CAMO','COD10K', 'NC4K']:
    for data_name in ['micai_te']:
        map_save_path = txt_save_path + "{}/".format(data_name)
        os.makedirs(map_save_path, exist_ok=True)
        evaluator(
            model=model,
            val_root='./dataset/micai/' + data_name + '/',
            # val_root='./dataset/TestDataset/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=384)
