# Vitali_Image_Style_Transfer
This deep learning project explores various architectures for image style transfer, making adjustments to study their effects and test potential improvements. The script used is based on rrmina's work, which can be found here: [rrmina/fast-neural-style-pytorch.](https://github.com/rrmina/fast-neural-style-pytorch/tree/master)

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


# Setup Instructions

## Step 1: Download Necessary Files

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

## Step 2: Create the right environment
#### Method 1:  Manual Setup

##### Create a New Environment: 
Install Python version 3.12.4 and create a new environment with the following command:
- `conda create -n myenv Python 3.12.4`
  
##### Activate the Environment:
- `conda activate myenv`

#### Install Required Libraries: 
Install the necessary libraries using `pip`:
- `pip install torch matplotlib pillow opencv-python numpy`


#### Method 2: Using an Existing Environment
If you encounter issues or prefer to use a pre-configured environment, you can set up the environment specified in the environment.yml file. This environment include additional libraries not required for the current script. To set up the environment, run:
- `conda env create -f environment.yml`
  
Make sure to activate the environment after creation:
- `conda activate myenv`

## Step 3: Run and Adjust the Script

Ensure the paths in your training script are updated to point to the downloaded files.  
Run the project in an environment with CUDA support for optimal performance.

### Run Training:
To initiate the training process, follow these steps:

1) **Run the Training Script:** Execute the file named Program.
2) **Configure Global Settings:**
   - Update the `DATASET_PATH`, `SAVE_PATH`, and `STYLE_IMAGE_PATH` variables with the correct file paths.
   - Adjust the `weight` and `ADAM_LR` variables according to your requirements.
3) **Seed Configuration:**
   - Modify or remove the `SEED` variable as needed. Keeping the same seed will result in similar initialization for different training runs, while changing it will introduce variability.

### Perform Style Transfer
Once you have trained and saved your models, or downloaded a pre-trained Transformer, you can perform style transfer on a single image or all images in a folder. To do this, follow these steps:
1) **Open the Style Transfer Script:** Execute the script named `Visualizzazione`
2) **Adjust Configuration:** Make sure to update the following settings as needed:
   - **Model Path:** Update the `STYLE_TRANSFORM_PATH` variable to the path of the model you wish to use.
   - **Color Preservation:** To retain the original colors of the content image, set `PRESERVE_COLOR=True`. Otherwise, set it to `False` 
   - **Select the Transformer Network:** Specify the network by setting net to the name of the transformer network you have trained. For example:
   - `net=transformer_norm.TransformerNetworkTanh`
   - `net=transformer.TransformerNetworkTanhWithAttention()`
3) **Input and Output Paths:**
   - When running the script, enter the path to the images you want to transform. If the images are in a subfolder relative to where you are running the script, you can specify the path as `subfolder/subfolder2`.
   - The script will create a subfolder named output within the specified path and save the transformed images there.
