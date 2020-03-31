## Description

Scripts to train and test CNN models using images and csv data.

### Inital Setup

Modify `config.ini` to set all parameters needed during training and testing.

    [IMAGES]
    images_dir = Path to the images directory, must contain train/val/test split sub-directories,
                 each split must have all images in folder/label mode.
    # Input size for the images into the network(s) used.
    width = Width size used to transform images.
    height = Height size usted to transform images.

    [CSV]
    csv_dir = Path to multimodal directory, must contain train/val/test csv files.
    csv_index = Index column used when loading csv dataframe, must match image names.

    [TRAINING]
    batch_size = Number of images used during each network forward pass.
    lr_rate = Initial learning rate used during training phase.
    epochs = Max number of epochs during training phase,
             training time is affected by this parameter and callbacks list in train.py
    cnn_network_list = List of networks to be trained,
                       supported networks are 'vgg16','vgg19','xception','resnet50', 'inceptionV3'
    # Optional parameter not implemented yet
    gpu_number = Optional parameter, used to train on multiple GPUs.

    [OUTPUT]
    logs_dir = Path to save tensorboard logs.
    models_dir = Path to save network models in .h5 format.
    figures_dir = Path to save confusion matrices obtained from test.py

### Usage

1. Install all packages in `requirements.txt`

2. Run `python train.py` to initiate the training phase.

3. Run `python test.py` to obtain results and confusion matrices from validation and test splits.
