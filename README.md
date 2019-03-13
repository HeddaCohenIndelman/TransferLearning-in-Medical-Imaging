# TransferLearning-in-Medical-Imaging
This is a case study of experimenting with Pneumonia detection in chest X-ray images of a public dataset

## Purpose
In this work I will try to classify X-ray images, taken from the dataset of Chest X-ray images, to Pneumonia/Normal. Such a framework can be adapted to other classiciation tasks in images with expected little effort.
## Data
Dataset is shared on https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/home
Dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal).
There are 5,863 X-ray images (JPEG) and 2 categories (Pneumonia/Normal).
The dataset should be downloaded and saved. Change the path if you intend to locate it on a different path then what I’ve specified.
### Acknowledgements
Data: https://data.mendeley.com/datasets/rscbjbr9sj/2
License: CC BY 4.0
Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

## Transfer Learning
In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pre-train a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest. 
These two major transfer learning scenarios look as follows:
1.	Fine-tuning the convent
Instead of random initialization, we initialize the network with a pre-trained network, like the one that is trained on ImageNet dataset. Rest of the training looks as usual.
2. ConvNet as fixed feature extractor
We will freeze the weights for all of the network except that of the final fully connected layer or layers. These last fully connected layers are replaced with a new layer or layers with random weights and only these layers are trained.
Code
I’ve written a python code (python 3.7) that implements these two transfer learning scenarios that may be used as a framework for similar projects.

See code Chest_XRay_Images_Pneumonia_classification.py

Its main parts include:
1.	Data augmentation and normalization for training and normalization for validation and test.
2.	Training with training and validation sets and evaluating on the test set based on the best performing model on the validation set.
3.	Fine-tuning the ConvNet was implemented as the last fully connected layer of a pre trained model of resnet18 was replaced with a new fully connected of output dimension of two to correspond to the two dimension of pneumonia versus normal. The entire neural network is then fine- tuned. You may see that there are total of 11180610 parameters and all of them are being trained.
4.	ConvNet as fixed feature extractor was implemented as the first N numbers (here, 4) were frozen and the rest of the model was fine tuned.
You may see that there are total of 11,180,610 parameters and 11,171,074 being trained.
5.	Early stopping in training when validation set loss has not improved for over a configurable number of epochs, as a mechanism to suppress over fitting.
6.	Plotting accuracy, cross entropy loss over epochs, ROC and confusion matrix.
7.	Trained model are saved for later usage.

A few comments
1.	This dataset has attracted much attention and it is not without flaw. Mainly, there is some dispute whether its labels are absolutely correct. Also, the predetermined training/ validation set split is very unorthodox. The validation set is only of 16 images which are too few for the task of estimating the performance using an independent validation set (held out set). 
It can be observed that accuracy and loss on the validation set varies dramatically from epoch to epoch and in general, does not seem to be a good approximation of the test set performance.
When constructing a real data set split, a common training/ validation set split is normally between 80%/ 2% and 60%/40% of data.
2.	Data augmentation can be extended, depending on the actual data.
3.	These main transfer learning paradigms can be extended. For example, set the early layers of a pre-trained network and add new layers which may be of other convolution types (e.g. depthwise convolution which may require less trainable parameters).
4.	Other performance criteria can be set, depending on the project goal (e.g emphasis certain types of errors with weighted cross entropy of set F-measure for harmonic mean of precision and recall). Here, we see that false positive error are far more common than false negative errors (more on this later).
5.	A Class Activation Mapping (CAM) may be implemented in order to highlight the areas in the images that triggered a Pneumonia prediction. With that, it can further assist in a diagnosis supportive system. 

## Results
When comparing the performance of the two models, one can see that though the performance during training of the first (Fine –tuning the ConvNet) seems better due to higher accuracy and lower loss, it in fact generalized worse as it achieved a lower accuracy on the test set.
Nevertheless, one can observe that the first model (Fine –tuning the ConvNet) achieved a much lower false-positive error (1/ 119) compared to that of the second model (11/ 11+144). It might be favorable in the context of diagnosis support system, where the model results might focus the radiology interpreter to examine some specific images with suspicious pathology.
1. Fine-tuning the ConvNet
![FineTune_Loss](https://github.com/HeddaCohenIndelman/TransferLearning-in-Medical-Imaging/blob/master/images/FineTuned_NN_Loss_over_epochs.png)
![FineTune_Accuracy](https://github.com/HeddaCohenIndelman/TransferLearning-in-Medical-Imaging/blob/master/images/FineTuned_NN_Accuracy_over_epochs.png)
![FineTune_ROC](https://github.com/HeddaCohenIndelman/TransferLearning-in-Medical-Imaging/blob/master/images/FineTunedROC.png)
![FineTune_CM](https://github.com/HeddaCohenIndelman/TransferLearning-in-Medical-Imaging/blob/master/images/FineTune_ConfusionMatrix.png)

2. 1.	ConvNet as fixed feature
![FixedFeature_Loss](https://github.com/HeddaCohenIndelman/TransferLearning-in-Medical-Imaging/blob/master/images/Pretrained%20NN_Loss_over_epochs.png)
![FixedFeature_Accuracy](https://github.com/HeddaCohenIndelman/TransferLearning-in-Medical-Imaging/blob/master/images/Pretrained%20NN_Accuracy_over_epochs.png)
![FixedFeature_ROC](https://github.com/HeddaCohenIndelman/TransferLearning-in-Medical-Imaging/blob/master/images/Pretrained_ROC.png)
![FixedFeature_CM](https://github.com/HeddaCohenIndelman/TransferLearning-in-Medical-Imaging/blob/master/images/Pretrained_ConfusionMatrix.png)




