import argparse 
import os 
import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.autograd import grad 
import torchvision 
from torchvision import datasets, transforms 
from torchvision.utils import save_image 
import torchvision.models as models 
import inversefed 
from utils.dataloader import DataLoader
from utils.stackeddata import StackedData

# inverting gradients algorithm from https://github.com/JonasGeiping/invertinggradients

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser(description='Adversarial attack from gradient leakage')
parser.add_argument('--model', type=str, help='model to perform adversarial attack')
parser.add_argument('--data', type=str, help='dataset used')
parser.add_argument('--stack_size', default=4, type=int, help='size use to stack images')
parser.add_argument('-l','--target_idx', nargs='+', help='list of data index to recontruct')
parser.add_argument('--save', type=str2bool, nargs='?', const=False, default=True, help='save')
parser.add_argument('--gpu', type=str2bool, nargs='?', const=False, default=True, help='use gpu')


args = parser.parse_args()
model_name = args.model
data = args.data
stack_size = args.stack_size
save_output = args.save 
if args.target_idx is not None: 
    target_idx = [int(i) for i in args.target_idx]
else: 
    target_idx = args.target_idx

device = 'cpu'
if args.gpu: 
    device = 'cuda'
print("Running on %s" % device)


def val_model(dataset, model, criterion):
    # evaluate trained model, record wrongly predicted index
    model.eval() 
    # record wrong pred index
    index_ls = [] 
    with torch.no_grad(): 
        val_loss, val_corrects = 0, 0 
        for batch_idx, (inputs, labels) in enumerate(dataset): 
            inputs = inputs.unsqueeze(dim=0).to(device)
            labels = torch.as_tensor([labels]).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0) # mutiply by number of batches
            val_corrects += torch.sum(preds == labels.data)
            if (preds != labels.data): 
                index_ls.append(batch_idx)
            if batch_idx == 100:
                break

        total_loss = val_loss / len(dataset) 
        total_acc = val_corrects.double() / len(dataset)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', total_loss, total_acc))
        return index_ls


dataloader = DataLoader(data, device)
dataset, data_shape, classes, (dm, ds) = dataloader.get_data_info() 
model = models.resnet18(pretrained=True) # use pretrained model from torchvision
model.fc = nn.Linear(512, len(classes)) # reinitialize model output: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
model = model.to(device)
model.eval()
criterion = nn.CrossEntropyLoss() 

stack_data = StackedData(stack_size=4, model_name=model_name, dataset_name=data, dataset=dataset, save_output=save_output, device=device)

if target_idx is None:
    wrong_pred_idx = val_model(dataset, model, criterion)
else:
    if isinstance(target_idx, (list))==False:
        wrong_pred_idx = [target_idx]
    else: 
        wrong_pred_idx = target_idx
    

stacked_data_d = stack_data.create_stacked_data(wrong_pred_idx)
for i in range(len(stacked_data_d['gt_img'])): 
    gt_img, gt_label, img_idx = stacked_data_d['gt_img'][i], stacked_data_d['gt_label'][i], stacked_data_d['img_index'][i]
    stack_pred = model(gt_img)
    target_loss = criterion(stack_pred, gt_label)
    input_grad = grad(target_loss, model.parameters())
    input_grad =[grad.detach() for grad in input_grad]
    # default configuration from inversefed
    config = dict(signed=True,
              boxed=False,
              cost_fn='sim',
              indices='def',
              norm='none',
              weights='equal',
              lr=0.1, 
              optim='adam',
              restarts=1,
              max_iterations=200,
              total_variation=0.1,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')
    
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=gt_img.shape[0])
    results = rec_machine.reconstruct(input_grad, gt_label, gt_img ,img_shape=data_shape)
    output_img, stats = results
    rec_pred = model(output_img)
    print('Predictions for recontructed images: ', [classes[l] for l in torch.max(rec_pred, axis=1)[1]])
    stack_data.grid_plot(img_idx, output_img, rec_pred, dm, ds)