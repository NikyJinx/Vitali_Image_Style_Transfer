# Vitali_Image_Style_Transfer
This deep learning project explores various architectures for image style transfer, making adjustments to study their effects and test potential improvements. The script used is based on rrmina's work, which can be found here: rrmina/fast-neural-style-pytorch.

Modifications Implemented:
- Adjusted the style and content weights used to calculate the weighted loss function.
- Introduced a linear variation in the weights during training (e.g., starting at 500 and decreasing to 50).
- Implemented a linear schedule for the learning rate during training.
- Experimented with different activation functions, such as SELU.
- Added dropout for regularization.
- Integrated an attention mechanism.

The original Transformer and TransformerNetworkTanh architectures were tested, with a detailed analysis of their respective issues and the conditions under which one outperforms the other.

Additionally, I experimented with different datasets beyond COCO, including:
  - Anime dataset 
  - Modified Face dataset
  - Landscape 
  - Personal photo

## Requirements: 

Most of the code requires a GPU with CUDA support. I used a 4060 Ti with 16GB of VRAM, which is highly recommended, especially when using the Transformer with the attention mechanism.


## Setup Instructions

### Step 1: Download Necessary Files

#### VGG16 Pre-trained Model
Download the pre-trained VGG16 model from this repository:  
[https://github.com/jcjohnson/pytorch-vgg](https://github.com/jcjohnson/pytorch-vgg)  
Place the file in the `content` folder, or adjust the file path in the training script to match the location of the downloaded file.

#### Pre-trained Transformers
Some transformer models have already been trained. To use these pre-trained weights, download the following file:  
[Pre-trained Transformers (Google Drive)](https://drive.google.com/file/d/1KonkFWUoCf-CyGY6HZ9Ea3bq703DHOO4/view?usp=drive_link)  
Extract the contents and place them in a folder such as `./content`. If you'd prefer to save the models in a different location, you can modify the file path in the script to match your saved location.

#### Pre-Processed Images
Some results have already been generated. To view these pre-processed images, download the following file:  
[Pre-Processed Images (Google Drive)](https://drive.google.com/file/d/1kcZeW-pgMJyBanEYk4ghpqE89x6h_ohQ/view?usp=drive_link)  
Extract the folder `./output`.

### Step 2: Adjust the Script

Ensure the paths in your training script are updated to point to the downloaded files.  
Run the project in an environment with CUDA support for optimal performance.
