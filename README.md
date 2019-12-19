# Introduction
Capsule network initially forked and adapted from Matwo -Caps-net git

## Train models
Run `main_train_and_test.py` to train the network. You can run the models by going into 'main_train_and_test.py' and uncommenting the associated dictionary parameters parameters at the bottom to get the apprpriate results. Please note that there is a augmentation json file "aug_dict_prob.json" which holds all the information pertaining how the augmentation parameters applied to each image. As a result this needs to be present in the same directory as main_train_and_test.py file. 

## Evaluate models
The following additional model scripts are used to assess and track the weights present in model. These scripts are the following and are ran in sequence:
-'get_wghts_sumary_stat.py': Pulls and plots all the different capsules weight distribution for each layer and produces summary statistics on each one. 
All other visualisation analysis performed with the SegCaps model was done using the testing suite present in the U-Net repository denoted as 'Masters-Thesis-UNet-repository' specific jupyter notebook Testing_dataset_unet.ipynb

## Train and test other datasets
In order to train and test on other datasets, modify the `dataset.py` file. See the example files and documentation for the specific file formats. Set the parameter `save_debug_images = True` in order to see, if the network input images are reasonable.

## Citation
If you use this code for your research, please cite our [Paper]():

```
@inproceedings{Bonheur2019,
  title     = {Matwo-CapsNet: a Multi-Label Semantic Segmentation Capsules Network},
  author    = {Bonheur, Savinien and {\v{S}}tern, Darko and Payer, Christian and Pienn, Michael and Olschewski, Horst and Urschler, Martin},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention - {MICCAI} 2019},
  doi       = {},
  pages     = {},
  year      = {2019},
}
```

## References
This code make use of the following projects:

[Framework](https://github.com/christianpayer/MedicalDataAugmentationTool)
```
@inproceedings{Payer2018a,
  title     = {Multi-label Whole Heart Segmentation Using {CNNs} and Anatomical Label Configurations},
  author    = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  booktitle = {Statistical Atlases and Computational Models of the Heart. ACDC and MMWHS Challenges. STACOM 2017},
  doi       = {10.1007/978-3-319-75541-0_20},
  pages     = {190--198},
  year      = {2018},
}
```

[Segcaps](https://github.com/lalonderodney/SegCaps)
```
@article{lalonde2018capsules,
  title={Capsules for Object Segmentation},
  author={LaLonde, Rodney and Bagci, Ulas},
  journal={arXiv preprint arXiv:1804.04241},
  year={2018}
}
```
## Special Requirement:

-cachetools 2.1.0
