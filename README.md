# Vitali_Image_Style_Transfer
This is a Deep Learning progect taht aims to test some architetture for image style transfer and make some change to study the effect and try some improvements. In particular i have use the script made by rrmina https://github.com/rrmina/fast-neural-style-pytorch/tree/master.

List of modification:
- Try different dataset other than coco:
  - Anime dataset 
  - Modified Face dataset
  - Some personal photo
- Changed the Style and Content Weight used to calculate the weighted Loss
- Added a linear change of the weights durng the training (e.g. starts at 500 finish at 50)
- Add a linear change of the learining rate during the training
- 

## Requirements: 

Most of the codes required a GPU with CUDAs, i used the 4060ti 16gb. The 16gb are highly suggest to run the scripts, evenmore if you'll use the Transformer with Attention Meccanism.

## Setup steps:

### Necessaries Donloads:


#### VGG16 pre-trained. https://github.com/jcjohnson/pytorch-vgg

Put this in the content folder, or in the training script change the location of the file to the one you prefer

#### Pre-train transformers. 
Some trains have been already done, to use the saved weight donload the folder content at this link: 

https://drive.google.com/file/d/1KonkFWUoCf-CyGY6HZ9Ea3bq703DHOO4/view?usp=drive_link

#### Pre-Processed Images.
Some result have been already processed, to see them download the folder output at this link: 

https://drive.google.com/file/d/1kcZeW-pgMJyBanEYk4ghpqE89x6h_ohQ/view?usp=drive_link 

