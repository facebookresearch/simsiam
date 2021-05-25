# SimSiam: Exploring Simple Siamese Representation Learning

<p align="center">
    <img width="400" alt="simsiam" src="https://user-images.githubusercontent.com/2420753/118343499-4c410100-b4de-11eb-9313-d49e65440a7e.png">
</p>

This is a PyTorch implementation of the [SimSiam paper](https://arxiv.org/abs/2011.10566):
```
@Article{chen2020simsiam,
  author  = {Xinlei Chen and Kaiming He},
  title   = {Exploring Simple Siamese Representation Learning},
  journal = {arXiv preprint arXiv:2011.10566},
  year    = {2020},
}
```

### Preparation

Install PyTorch and download the ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Similar to [MoCo](https://github.com/facebookresearch/moco), the code release contains minimal modifications for both unsupervised pre-training and linear classification to that code. 

In addition, install [apex](https://github.com/NVIDIA/apex) for the [LARS](https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py) implementation needed for linear classification.

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
The script uses all the default hyper-parameters as described in the paper, and uses the default augmentation recipe from [MoCo v2](https://arxiv.org/abs/2003.04297). 

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

Our pre-trained ResNet-50 models and logs:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">batch<br/>size</th>
<th valign="bottom">pre-train<br/>ckpt</th>
<th valign="bottom">pre-train<br/>log</th>
<th valign="bottom">linear cls.<br/>ckpt</th>
<th valign="bottom">linear cls.<br/>log</th>
<th valign="center">top-1 acc.</th>
<!-- TABLE BODY -->
<tr>
<td align="center">100</td>
<td align="center">512</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar">link</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/simsiam/logs/100ep/pretrain.log">link</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/simsiam/models/100ep/linear/model_best.pth.tar">link</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/simsiam/logs/100ep/linear.log">link</a></td>
<td align="center">68.1</td>
</tr>
<tr>
<td align="center">100</td>
<td align="center">256</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar">link</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/simsiam/logs/100ep-256bs/pretrain.log">link</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/linear/model_best.pth.tar">link</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/simsiam/logs/100ep-256bs/linear.log">link</a></td>
<td align="center">68.3</td>
</tr>
</tbody></table>

Settings for the above: 8 NVIDIA V100 GPUs, CUDA 10.1/CuDNN 7.6.5, PyTorch 1.7.0.

### Transferring to Object Detection

Same as [MoCo](https://github.com/facebookresearch/moco) for object detection transfer, please see [moco/detection](https://github.com/facebookresearch/moco/tree/master/detection).


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.