# SimSiam: Exploring Simple Siamese Representation Learning

This is a PyTorch implementation of the [SimSiam paper](https://arxiv.org/abs/2011.10566):
```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Kaiming He},
  title   = {Exploring Simple Siamese Representation Learning},
  journal = {arXiv preprint arXiv:2011.10566},
  year    = {2020},
}
```

### Preparation

Install PyTorch and download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Similar to [MoCo](https://github.com/facebookresearch/moco), the code release contains minimal modifications for both unsupervised pre-training and linear classification to that code. 

In addition, install [apex](https://github.com/NVIDIA/apex) for the [LARS](https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py) implementation needed for linear classifier training.

### Unsupervised Pre-Training

Only **multi-gpu**, **DistributedDataParallel** training is supported; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_simsiam.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr \
  [your imagenet-folder with train and val folders]
```
The script uses all the default hyper-parameters as described in the paper, and uses the default augmentation recipe from MoCo v2(https://arxiv.org/abs/2003.04297). 

The above command performs pre-training with a non-decaying predictor learning rate for 100 epochs, corresponding to the last row of Table 1 in the paper. 

### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/checkpoint_0099.pth.tar \
  --lars \
  [your imagenet-folder with train and val folders]
```

The above command uses LARS optimizer and a default batch size of 4096.

### Models and Logs

Our pre-trained ResNet-50 models and logs with 8 NVIDIA V100 GPUs, CUDA 10.2:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">pre-train<br/>ckpt</th>
<th valign="bottom">pre-train<br/>log</th>
<th valign="bottom">linear cls.<br/>ckpt</th>
<th valign="bottom">linear cls.<br/>log</th>
<th valign="center">top-1 acc.</th>
<!-- TABLE BODY -->
<tr>
<td align="center">100</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar">link</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/simsiam/logs/100ep/pre-train.log">link</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/simsiam/models/100ep/finetune/model_best.pth.tar">link</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/simsiam/logs/100ep/finetune.log">link</a></td>
<td align="center">68.1</td>
</tr>
</tbody></table>


### Transferring to Object Detection

Same as [MoCo](https://github.com/facebookresearch/moco) for object detection transfer, please see [moco/detection](https://github.com/facebookresearch/moco/tree/master/detection).


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.