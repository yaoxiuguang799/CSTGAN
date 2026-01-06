# CSTGAN
cascaded spatiotemporal generative adversarial network

## Usage Instructions for Reproducibility

Usage Instructions for Reproducibility

This section provides step-by-step instructions to reproduce the experimental results reported in the paper using CSTGAN.

1. Environment Setup
   
Clone or download the CSTGAN repository to your local machine:

git clone <repository_url>
cd CSTGAN

Install the required dependencies:

PyTorch

NumPy

logging

tsdb

h5py

Example installation command:

pip install torch numpy tsdb h5py


⚠️ Please ensure that the installed PyTorch version is compatible with your CUDA environment.

2. Model Training

Download the training dataset and place it in the following directory:

./data/dataset


Dataset download link:
https://pan.baidu.com/s/1S9HU2lCgJmcMaenbQLroDQ?pwd=82g9

Open the main script main.py and set:

is_Train = True


Start the training process:

python main.py


After training, the learned model parameters will be automatically saved to the predefined output directory.

3. Model Evaluation

Download the validation dataset and place it in the same directory:

./data/dataset


Dataset download link:
https://pan.baidu.com/s/1S9HU2lCgJmcMaenbQLroDQ?pwd=82g9

Modify the configuration in main.py as follows:

is_Train = False
trained_model_path = "path/to/trained_model.pth"


Where:

trained_model_path can be set to a model trained by yourself

Alternatively, you may use the pre-trained model provided by the authors

Dataset download link:
https://pan.baidu.com/s/1S9HU2lCgJmcMaenbQLroDQ?pwd=82g9

Run the evaluation:

python main.py

4. Notes

Training and evaluation share the same data directory structure

Please ensure that dataset paths and model paths are correctly configured

For exact reproducibility, we recommend using the same hyperparameter settings as described in the paper

