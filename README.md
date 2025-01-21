# Generative_Image_Rotation (Protoplanetary Disk Image Rotation using Pix2Pix cGAN)

### NOTE : Kindly install the dependencies by running the following code line:
** pip install -r requirements.txt **

### Overview:
This repository hosts Jupyter notebooks dedicated to training and inference of a Conditional Generative Adversarial Network (cGAN) based on the Pix2Pix architecture. The project focuses on generating standardized face-on images of Protoplanetary Disks from source images that are randomly oriented, using a dataset obtained from FARGO3D simulations.

### Key Features:
- **Training Notebook:**
  - Implements a cGAN model with Pix2Pix architecture to transform randomly oriented Protoplanetary Disk images into face-on views.
  - Includes attention mechanisms to enhance image generation quality by focusing on crucial features.
  - Utilizes TensorFlow for model development and training.

- **Inference Notebook:**
  - Deploys the trained cGAN model to generate face-on images from new, randomly oriented Protoplanetary Disk images.
  - Enables comparison between generated images and corresponding target images for evaluation.

- **Dataset Handling:**
  - Preprocesses and stores Protoplanetary Disk images in numpy arrays for efficient memory usage during training.
  - Includes 21,000 images sourced from FARGO3D simulations, comprising 30 orientations per image.

- **Performance Evaluation:**
  - Incorporates periodic model checkpoints and performance visualizations using `summarize_performance` function.
  - Automatically organizes saved models and plots into uniquely named folders for each training session.


### Supervisor:
Dr. Sayantan Auddy 

### Author:
Dyutiman Santra
