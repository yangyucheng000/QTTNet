# QTTNet-Quantized Tensor Train Neural Networks for 3D Object and Video Recognition

## 介绍

This is code for the implementation of QTTNet in Mindspore.

## Data&Preparation

Note: This part of the code cannot pass the gitee_gate code check temporarily, and will be submitted later. As an alternative we provide a simple implementation on MNIST as a demo.

Download the ModelNet and UCF on official link

```ruby
    ModelNet: <https://modelnet.cs.princeton.edu/>
    UCF: <https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php>
```

To prepare dataset for training QTTNet, you should run py files by order.

ModelNet

```ruby
    'QTTNet/ModelNet40/data/convert_modelnet40.py': Convert npy.tar from official voxelized data(.mat)
    'QTTNet/ModelNet40/data/data_loader.py': Prepare data(.h5) for training QTTNet & Augmentation
    'QTTNet/ModelNet40/data/h5.py': Convert to h5 files to tfrecord
```

UCF

```ruby
    'QTTNet/UCF/PrepareData.py': Read video data and prepare material for training QTTNet
    'QTTNet/UCF/InputData.py': Input data to QTTNet & Augmentation
```

## Experiment

ModelNet40

```ruby
    python Modelnet_main.py --data_dir '/data_dir/' --model_dir '/model_dir/'
```

UCF

```ruby
    python Action3DCNN.py flag_log_dir '/log_dir/'
```
