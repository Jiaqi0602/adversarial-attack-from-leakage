# From Gradient Leakage to Adversarial Attacks in Federated Learning

By utilizing an existing privacy
breaking algorithm which inverts gradients of models to reconstruct the input data, the data reconstructed from inverting gradients algorithm reveals the vulnerabilities of models in representation learning.

In this work, we utilize the inverting gradients algorithm proposed in [Inverting Gradients - How easy is it to break Privacy in Federated Learning?](https://arxiv.org/pdf/2003.14053.pdf) to reconstruct the data that could lead to possible threats in classification task. By stacking one wrongly predicted image into different batch sizes, then use the stacked images as input of the existing gradients inverting algorithm will result in reconstruction of distorted images that can be correctly predicted by the attacked model.

<<<<<<< HEAD
![demo](image/rec_output.jpg)
![graph1](graph1.jpg) ![graph2](data/graph2.jpg)

=======
<p align="center">
  <img src="https://raw.githubusercontent.com/Jiaqi0602/adversarial-attack-from-leakage/main/image/rec_output.JPG" width="400" height="400" align="left"/>
  <img src="https://raw.githubusercontent.com/Jiaqi0602/adversarial-attack-from-leakage/main/image/graph1.jpg" height="200" width="385" align="right"/>
  <img src="https://raw.githubusercontent.com/Jiaqi0602/adversarial-attack-from-leakage/main/image/graph2.jpg" height="200" width="385" align="right"/>
</p>
>>>>>>> 9de60a6... modify README

## Prerequisites
Required libraries:
```bash
Python>=3.7
pytorch=1.5.0
torchvision=0.6.0
```
## Code
```python
python main.py --model "resnet18" --data "cifar10" stack_size 4 -ls 1001,770,123 --save True --gpu True
```

Implementation for ResNet-18 trained with CIFAR10 can be found [HERE](https://github.com/Jiaqi0602/adversarial-attack-from-leakage/blob/main/demo%20-%20CIFAR10.ipynb) and with VGGFACE2 can be found [HERE](https://github.com/Jiaqi0602/adversarial-attack-from-leakage/blob/main/demo%20-%20VGGFACE2.ipynb)

#### Quick reproduction for CIFAR10 dataset: 
You can download pretrained model from [HERE](https://github.com/huyvnphan/PyTorch_CIFAR10) then replace the torchvision models.


## Reference: 
- [Inverting Gradients - How easy is it to break Privacy in Federated Learning?](https://github.com/JonasGeiping/invertinggradients)
- [Deep Leakage From Gradients](https://github.com/mit-han-lab/dlg) 
- [PyTorch models trained on CIFAR-10 dataset](https://github.com/huyvnphan/PyTorch_CIFAR10)
