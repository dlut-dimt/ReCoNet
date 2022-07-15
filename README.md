# ReCoNet

![visitors](https://visitor-badge.glitch.me/badge?page_id=MisakiCoca.ReCoNet)

Zhanbo Huang, Jinyuan Liu, Xin Fan*, Risheng Liu, Wei Zhong, Zhongxuan Luo.
**"Recurrent Correction Network for Fast and Efficient Multi-modality Image Fusion"**, European Conference on Computer
Vision **(ECCV)**, 2022.

## Milestone

In the near future, we will publish the following materials.

* v0 [ECCV]: Fuse network (ReCo) with pre-trained parameters for generating results in paper. **Finished**
* v1: A new script & architecture of ReCo+ for fast training & prediction. **Building**
* v1: A highly robust pre-trained parameters for ReCo+ based on realistic scene training. (We are collecting data with
  realistic implications.)

## Update

[2022-07-15] Train script for ReCo(v1) is available!

[2022-07-13] Preview of micro-register is available!

[2022-07-12] The ReCo(v0) is available!

## Requirements

* Python 3.10
* PyTorch 1.12
* TorchVision 0.13.0
* PyTorch lightning 0.8.5
* Kornia 0.6.5

## Extended Experiments

### Generate fake visible images

To generating fake visible images as described in our paper, you can refer to my
another repository [complex-deformation](https://github.com/MisakiCoca/complex-deformation), which is a component of
this work.

It shows how we can deform the image and generate a restored field that **approximates** the ground truth.

### Have a quick preview of our micro-register

To give a quick preview of our micro-register module, you can try the training & prediction based on
the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

Activate your conda environment and enter folder `exp/test_register`.

1. To train the register yourself, you just need to run this code.

```shell
export PYTHONPATH="${PYTHONPATH}:$RECO_ROOT"
python test_register.py --backbone $BACKBONE --dst $DST
```

The `$RECO_ROOT` is the root path of ReCo repository, like `~/lab/reco`, the `$BACKBONE` denotes which architecture to
use `m`-`micro` or `u`-`unet`.

We will do following things automatically: download MNIST dataset, train the register, and save predictions in `$DST`.

2. If you just want to test the performance, we offer pre-trained parameters for both `micro` and `unet` based register.

```shell
export PYTHONPATH="${PYTHONPATH}:$RECO_ROOT"
python test_register.py --backbone $BACKBONE --dst $DST --only_pred
```

The prediction results will be save in `$DST` and the patches from left to right are `moving`, `fixed` and `moved`,
respectively.

## Get start (v0) (**Recommended for Now**)

1. To use our pre-trained parameters of ECCV-22 for fusion, you need to prepare your dataset in `$ROOT/data/$NAME`.

```
  $DATA (dataset name, like: tno)
  ├── ir
  ├── vi
```

2. Enter the archive folder `cd archive`, and activate your conda environment `conda activate $CONDA_ENV`.

```shell
export PYTHONPATH="${PYTHONPATH}:$RECO_ROOT"
python fuse.py --ir ../data/$DATA/ir --vi ../data/$DATA/vi --dst $SAVE_TO_WHERE 
```

3. Now, you will find the fusion results in `$SAVE_TO_WHERE`, this operation will create output folder automatically.

## Get start (v1) **Preview Version**

**Only recommended if you are intending in training ReCo+ yourself.**

**Note that: Due to the instability of the micro-register module in the future, we recommend training only the fusion
part.**

1. To use the script to train ReCo+ yourself, you need to prepare your dataset in `$ROOT/data/$NAME`.

```
  $DATA (dataset name, like: tno)
  ├── ir
  ├── vi
  ├── iqa (new for v1, optional)
  |   |   ├── ir (information measurement for infrared images)
  |   |   ├── vi (information measurement for visible images)
  ├── meta (new for v1)
  |   |   ├── train.txt (which images are used for training)
  |   |   ├── val.txt (which images are used for validation)
  |   |   ├── pred.txt (which images are used for prediction)
```

2. Activate your conda environment `conda activate $CONDA_ENV`.

```shell
# set project path for python
export PYTHONPATH="${PYTHONPATH}:$RECO_ROOT"
# only train fuse part (ReCo) **current recommended**
python train.py --data data/$DATA --ckpt $CHECKPOINT_PATH --lr 1e-3
# train registration and fuse (ReCo+)
python train.py --register m --data data/$DATA --ckpt $CHECKPOINT_PATH --lr 1e-3 --deform $DEFORM_LEVEL
```

The `$DEFORM_LEVEL` should be `easy`, `normal` or `hard`.

⚠️ Limitations: As mentioned in the paper, when the difference between mid-wave infrared and visible images in your
dataset is too large, the register may not converge properly.

3. To generate the fusion images with pre-trained parameters, just run the following.

```shell
# set project path for python
export PYTHONPATH="${PYTHONPATH}:$RECO_ROOT"
# only fuse part (ReCo) **current recommended**
python pred.py --data $data/$DATA --ckpt $CHECKPOINT_PATH --dst $SAVE_TO_WHERE
# registration & fuse (ReCo+)
 python pred.py --register m --data $data/$DATA --ckpt $CHECKPOINT_PATH --dst $SAVE_TO_WHERE
```
