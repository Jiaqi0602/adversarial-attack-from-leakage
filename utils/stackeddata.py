import os 
import numpy as np 
import matplotlib.pyplot as plt 
import json
import torch
from inversefed import consts
from torchvision import datasets, transforms 
from torchvision.utils import save_image

class StackedData: 
    def __init__(self, stack_size, model_name, dataset_name, dataset, save_output, device): 
        self.stack_size = stack_size
        self.model_name = model_name 
        self.dataset_name = dataset_name 
        self.dataset = dataset
        self.save_output = save_output 
        self.device = device 
        
    def create_stacked_data(self, index_ls): 
        batch_data = {'gt_img': [], 'gt_label': [], 'img_index': []} 
        for index in index_ls: 
            gt_img, gt_label = self.dataset[index]
            gt_images, gt_labels = [], [] 
            for i in range(self.stack_size): 
                gt_images.append(gt_img) 
                gt_labels.append(torch.as_tensor((gt_label,), device=self.device))

            gt_images_ = torch.stack(gt_images).to(self.device)
            gt_labels_ = torch.cat(gt_labels)
            batch_data['gt_img'].append(gt_images_)
            batch_data['gt_label'].append(gt_labels_)
            batch_data['img_index'].append(index)

        return batch_data

    def grid_plot(self, img_idx, tensors, logit, dm, ds):
        _, indices = torch.max(logit, 1)
        accuracy, _ =  torch.max(torch.softmax(logit, dim=1).cpu(), dim=1)
        labels = list(zip(indices.cpu().numpy(), list(np.around(accuracy.detach().numpy(),4))))
        
        # un-normalize before plotting 
        tensors = tensors.clone().detach()
        tensors.mul_(ds).add_(dm).clamp_(0, 1)

        if self.save_output: 
            if os.path.exists('output_rec_images')==False: 
                os.makedirs('output_rec_images')

            if self.model_name is None: 
                saved_name = '{}_{}_{}'.format(self.dataset_name, self.stack_size, img_idx)
            else: 
                saved_name = '{}_{}_{}_{}'.format(self.model_name, self.dataset_name, self.stack_size, img_idx)

            for i, tensor in enumerate(tensors): 
                extension = '.png'
                saved_name_ = "{}_{}_{}{}".format(saved_name, self.dataset.classes[indices[i]], i, extension) 
                save_image(tensor, os.path.join('output_rec_images', saved_name_))

        if tensors.shape[0]==1: 
            tensors = tensors[0]
            plt.figure(figsize=(4,4))
            plt.imshow(tensors.permute(1,2,0).cpu())
            plt.title(self.dataset[labels])

        else: 
            grid_width = int(np.ceil(len(labels)**0.5))
            grid_height = int(np.ceil(len(labels) / grid_width))

            fig, axes = plt.subplots(grid_height, grid_width, figsize=(3, 3))
            for im, l, ax in zip(tensors, labels, axes.flatten()):
                ax.imshow(im.permute(1, 2, 0).cpu());
                ax.set_title(l)
                ax.axis('off')
        plt.show() 
    