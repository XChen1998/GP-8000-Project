
# GP8000 Group Project

This repository contains the GP8000 group project, which uses PyTorch to implement ResNet models for classifying brain tumour MRI images.

## Setup Instructions

### 0. Create a Conda Environment

To set up the environment, follow these steps:

1. **Create a new Conda environment**:
    ```bash
    conda create --name GP8000 python=3.9
    ```

2. **Activate the Conda environment**:
    ```bash
    conda activate GP8000
    ```

3. **Install the required dependencies**:    
    ```bash
    pip install -r requirements.txt
    ```

Once the environment is set up, you can use any Jupyter notebook-compatible IDE (such as VSCode) with the installed kernel to run the code.

### 1. Download the Dataset

Please download the brain tumour MRI dataset from this [Kaggle link](https://www.kaggle.com). Follow the instructions provided on the website to obtain the dataset.

After downloading, save all data to the same directory as `preprocessing.py`.

### 2. Data Preprocessing

You need to preprocess the data using the `preprocessing.py` script. Follow one of these methods:

- **Method 1**: Run the script in a scientific computing IDE (e.g., Spyder) with the Conda kernel.
- **Method 2**: Run the script directly from the command line:

    ```bash
    python preprocessing.py
    ```

This will generate a folder called `cleaned` in the same directory, containing the normalised data. This step is essential as it prepares the data for training.

### 3. Model Training

The file `model.py` contains a PyTorch implementation of the ResNet family, created from scratch. You can modify the number of layers in the section labelled `# The model architecture` to adjust the depth of the ResNet model.

Training logs and results, will be automatically saved in a folder named: `ResNet-{depth}_TumorClassification_logs`.

### 4. Confusion Matrix and Predictions

The code also includes blocks for generating a confusion matrix and visualising predicted samples after the model has been trained.

### 5. Analysis

There are several other intuitive and easy-to-use Python scripts to make plots.

### Hardware Requirements

- **GPU**: You must have a valid NVIDIA GPU that supports PyTorch hardware acceleration, with at least 16 GiB of video RAM.
- **System RAM**: At least 16 GiB of system RAM is required.

### Contact Information

For further queries, please contact the author: [xinghe001@e.ntu.edu.sg](mailto:xinghe001@e.ntu.edu.sg)
