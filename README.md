# Neuron Attribution-based Attacks
Official Tensorflow implementation for "Improving Adversarial Transferability via Neuron Attribution-based Attacks" (CVPR 2022).

**[Improving Adversarial Transferability via Neuron Attribution-Based Attacks](https://arxiv.org/pdf/2204.00008.pdf)  (CVPR 2022)**

## Requirements

- Python 3.6.8
- Keras 2.2.4
- Tensorflow 1.14.0
- Numpy 1.16.2
- Pillow 6.0.0
- Scipy 1.2.1

## Experiments

You should download the  pretrained models from ( https://github.com/tensorflow/models/tree/master/research/slim, and https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models) before running the code. Then place these model checkpoint files in `./models_tf`.

#### Introduction


- `NAA.py` : the implementation for NAA attack.

- `attacks.py` : the implementation for NAA attack and baseline attacks (NRDM, FDA, FIA).

- `verify.py` : the code for evaluating generated adversarial examples on different models.

#### Layer setting

- inception_v3: InceptionV3/InceptionV3/Mixed_5b/concat

- inception_v4: InceptionV4/InceptionV4/Mixed_5e/concat

- inception_resnet_v2: InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu

- resnet_v2_152: resnet_v2_152/block2/unit_8/bottleneck_v2/add
  

#### Example Usage

##### Generate adversarial examples:

- NAA

```
python NAA.py --model_name inception_v3 --attack_method NAA --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --ens 30 --output_dir ./adv/NAA/
```

- NAA-PD

```
python NAA.py --model_name inception_v3 --attack_method NAAPIDI --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --ens 30 --amplification_factor 2.5 --gamma 0.5 --Pkern_size 3 --prob 0.7 --output_dir ./adv/NAAPIDI/
```

- PIM:

```
python NAA.py --model_name inception_v3 --attack_method PIM --amplification_factor 2.5 --gamma 0.5 --Pkern_size 3 --output_dir ./adv/PIM/
```

- NRDM

```
python attacks.py --model_name inception_v3 --attack_method NRDM --layer_name InceptionV3/InceptionV3/Mixed_5b/concat --output_dir ./adv/NRDM/
```

Attack methods have different parameter setting for different source models, and the detailed setting can be found in our paper.

##### Evaluate the attack success rate

```
python verify.py --ori_path ./dataset/images/ --adv_path ./adv/NAA/ 
```

## Citing this work

If you find this work is useful in your research, please consider citing:

```
@inproceedings{zhang2022improving,
  title={Improving adversarial transferability via neuron attribution-based attacks},
  author={Zhang, Jianping and Wu, Weibin and Huang, Jen-tse and Huang, Yizhan and Wang, Wenxuan and Su, Yuxin and Lyu, Michael R},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14993--15002},
  year={2022}
}
```

## Acknowledgments

Code refer to: [Feature Importance-aware Attack](https://github.com/hcguoO0/FIA)
