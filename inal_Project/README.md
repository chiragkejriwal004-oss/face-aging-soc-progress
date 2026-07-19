# Face Aging using Deep Learning (SoC Final Project)

**Author:** Chirag Kejriwal  
**Institution:** Indian Institute of Technology (IIT) Bombay, Department of Civil Engineering  
**Program:** Summer of Core (SoC) - Deep Learning and AI  

## Project Overview
This repository contains the final project for the Summer of Core (SoC) Deep Learning track. The objective is to perform genuine AI-based age progression on facial images. Rather than relying on simple 2D filters, this project utilizes a pre-trained Generative Adversarial Network (GAN) to encode a face into a latent space and manipulate its age attributes photorealistically.

## Objectives
1. Load and explore a standard facial dataset (UTKFace) from Kaggle.
2. Implement a pre-trained Deep Learning model for age transformation.
3. Process custom user images, apply the AI model, and visualize side-by-side comparisons.

## Deep Learning Concepts Used
* **Generative Adversarial Networks (GANs):** Specifically, the architecture builds upon StyleGAN2.
* **Latent Space Embedding (pSp):** The model uses a pixel2style2pixel (pSp) encoder to translate real-world images into the StyleGAN latent space.
* **Feature Manipulation:** Altering the specific vectors in the latent space that correspond to the "age" feature without losing the core "identity" features.

## Dataset Description
**UTKFace:** A large-scale facial dataset consisting of over 20,000 images with annotations for age (0 to 116 years), gender, and ethnicity. It serves as an excellent benchmark dataset for understanding how deep learning models map human facial features across different demographics.

## AI Model Used
**SAM (Style-Based Age Manipulation):**  
Developed by Yuval Alaluf et al., SAM is a state-of-the-art framework for lifespan age transformation. It was chosen over simple OpenCV filters because it performs a structural, 3D-aware age progression that considers skin elasticity, hair thinning, and bone structure changes via StyleGAN2.

## Methodology
1. **Environment Setup:** Configured Google Colab with required libraries (PyTorch, dlib, OpenCV).
2. **Model Loading:** Cloned the official SAM repository and downloaded the pre-trained weights (`sam_ffhq_aging.pt`).
3. **Preprocessing:** Resized and normalized the input image to a $256 \times 256$ tensor required by the pSp encoder.
4. **Inference:** Passed the tensor through the model to extract latents and regenerate the aged image.
5. **Post-processing:** Denormalized the output tensor back into a standard RGB PNG format.

## Installation & Execution Instructions
To run this project:
1. Open `Face_Aging_Project.ipynb` in Google Colab.
2. Ensure the Runtime is set to **GPU** (`Runtime` -> `Change runtime type` -> `T4 GPU`).
3. Run the cells sequentially from top to bottom.
4. When prompted by Cell 5, upload a clear, front-facing cropped photo of a face (e.g., `test_face.jpg`).
5. The notebook will automatically apply the model and save the outputs to the `output/` directory.

## Results
The model successfully extracts the facial identity and applies synthetic aging. 
* Outputs are stored in the `output/` folder.
* **`comparison.png`** showcases the side-by-side visualization of the original versus the aged face.

## Future Improvements
* Integrate `dlib` face-landmark detection to automatically crop and align uncentered, raw images before feeding them to the encoder.
* Implement a slider widget in Colab to allow the user to select specific target ages (e.g., 50, 70, 90 years old) rather than a static forward pass.

## References
* Alaluf, Y., Patashnik, O., & Cohen-Or, D. (2021). *Only a Matter of Style: Age Transformation Using a Style-Based Regression Model.* ACM SIGGRAPH.
* Karras, T., et al. (2020). *Analyzing and Improving the Image Quality of StyleGAN.* CVPR.
