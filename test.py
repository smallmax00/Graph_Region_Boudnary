import torch.nn.functional as F
import numpy as np
import torch
import os
import argparse
from .lib.model import ODOC_seg_edge_gru_gcn
from .utils.Dataloader_ODOC import ODOC
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='path-to-your-data', help='Name of Experiment')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
saved_model_path = os.path.join(snapshot_path, 'best_model.pth')

if __name__ == "__main__":
    model = ODOC_seg_edge_gru_gcn()
    model = model.cuda()

    db_test = ODOC(base_dir=train_data_path, split='test')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    best_performance = 0.0
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()
    with torch.no_grad():
        for i_batch, (sampled_batch, sampled_name) in enumerate(testloader):
            volume_batch, label_batch, edge_batch = sampled_batch['img'], sampled_batch['mask'], sampled_batch['con_gau']
            volume_batch, label_batch, edge_batch = volume_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor), edge_batch.type(torch.FloatTensor)
            volume_batch, label_batch, edge_batch = volume_batch.cuda(), label_batch.cuda(), edge_batch.cuda()

            outputs1, edge_outputs1, g_edge_1 = model(volume_batch)
            pred_edge = F.upsample(input=edge_outputs1, size=(256, 256), mode='bilinear')
            g_edge_output = F.upsample(input=g_edge_1, size=(256, 256), mode='bilinear')
            pred_seg = F.upsample(input=outputs1, size=(256, 256), mode='bilinear')
            # seg
            y_pre = pred_seg.cpu().data.numpy().squeeze()
            y_pre_gt = label_batch.cpu().data.numpy().squeeze()


            y_map_cup = (y_pre[0] > 0.5).astype(np.uint8)
            y_map_disc = (y_pre[1] > 0.5).astype(np.uint8)

            """
            "uncomment below if a smoother boundary"
            
            image = Image.fromarray(y_map_cup)
            filter_image = image.filter(ImageFilter.ModeFilter(size=20))
            y_map_cup = np.asarray(filter_image)
            y_map_cup = (y_map_cup > 0).astype(np.uint8)

            image = Image.fromarray(y_map_disc)
            filter_image = image.filter(ImageFilter.ModeFilter(size=20))
            y_map_disc = np.asarray(filter_image)
            y_map_disc = (y_map_disc > 0).astype(np.uint8)
            """

            # edge
            y_pre_edge = pred_edge.cpu().data.numpy().squeeze()
            y_pre_edge_gt = edge_batch.cpu().data.numpy().squeeze()
            y_edge_cup = (y_pre_edge > 0.5).astype(np.uint8)
            y_edge_disc = (y_pre_edge > 0.5).astype(np.uint8)

            """Uncomment below if a smoother boundary
            # image = Image.fromarray(y_edge_cup)
            # filter_image = image.filter(ImageFilter.ModeFilter(size=30))
            # y_map_cup = np.asarray(filter_image)
            # y_edge_cup = (y_edge_cup > 0).astype(np.uint8)
            
            # image = Image.fromarray(y_edge_disc)
            # filter_image = image.filter(ImageFilter.ModeFilter(size=30))
            # y_edge_disc = np.asarray(filter_image)
            # y_edge_disc = (y_edge_disc > 0).astype(np.uint8)
            """

            plt.figure()
            plt.subplot(1, 4, 1)
            plt.imshow(y_edge_cup, cmap='gray')
            plt.subplot(1, 4, 2)
            plt.imshow(y_edge_disc, cmap='gray')
            plt.subplot(1, 4, 2)
            plt.imshow(y_map_cup, cmap='gray')
            plt.subplot(1, 4, 2)
            plt.imshow(y_map_disc, cmap='gray')
            plt.show()








