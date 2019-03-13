# -*- coding: utf-8 -*-
"""
In this work I will try to classify X-ray images, taken from
the dataset of Chest X-ray images, to Pneumonia/Normal.
Dataset is shared on https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/home
Dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal).
There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Acknowledgements
Data: https://data.mendeley.com/datasets/rscbjbr9sj/2

License: CC BY 4.0

Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

Quoting these notes from cs231n:,

    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.

These two major transfer learning scenarios look as follows:

-  **Finetuning the convnet**: Instead of random initializaion, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer or layers. These last fully connected layers are replaced with a new layer or layers
   with random weights and only these layers are trained.

"""

# Author: Hedda Cohen

#from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from torch.autograd import Variable
import mlxtend
#my plotting module
import graphs_plotter


plt.ion()   # turn on interactive mode

def label_counter_in_dataset(dataloader):
    # Get the counts for each class
    normal_cases = 0
    for data, labels in dataloader:
        normal_cases += len(torch.nonzero(labels))
    return normal_cases

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #Normalize a tensor image with mean and standard deviation.
        # Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel of the input torch.*Tensor
        # i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]
        #Convert a color image to grayscale and normalize the color range to [0,1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}

# Visualize a few images
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.imsave('grid display', inp)

# Trains a model. We will schedule the learning rate using torch.optim.lr_scheduler and save the best performing model on the validation set.
# Notice early stopping which halts the training when the validation loss has not decreased for a number of epochs, to prevent overfitting
def train_model(model, criterion, optimizer, scheduler, num_epochs = 35, max_epochs_stop = 10, model_name ='direct_copy_models'):
    since = time.time()
    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Early stopping intialization
    epochs_no_improve = 0

    loss_train, acc_train = [], []
    loss_val, acc_val = [], []
    epochs_array = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                epochs_array.append(epoch)
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = Variable(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history only while training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                loss_train.append(epoch_loss)
                acc_train.append(epoch_acc.item())
            else:
                loss_val.append(epoch_loss)
                acc_val.append(epoch_acc.item())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                # option1
                #torch.save(model.state_dict(), os.path.join(root_dir, 'models'))
                #option 2
                torch.save(model, os.path.join(root_dir, model_name))
                #best_model_wts = copy.deepcopy(model.state_dict())

                # Track improvement
                epochs_no_improve = 0
                best_acc = epoch_acc

            # Otherwise increment count of epochs with no improvement
            elif phase == 'val' and epoch_acc <= best_acc:
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print("Early Stopping! Total epochs:",epoch, "validation loss ", epoch_loss, "validation acc", epoch_acc.item(), "best acc", best_acc.item() )
                    # Load the best state dict
                    torch.load(os.path.join(root_dir, model_name))
                    time_elapsed = time.time() - since
                    return best_acc, epochs_array, acc_val, acc_train, loss_val, loss_train, time_elapsed

    time_elapsed = time.time() - since
    return best_acc, epochs_array, acc_val, acc_train, loss_val, loss_train, time_elapsed

# Tests the trained model
def test_model(model):
    print("testing"+'\n')
    since = time.time()

    acc_test = []
    running_corrects = 0
    model.eval()  # Set model to evaluate mode

    # Iterate over data.
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)

    acc_val = running_corrects.double() / dataset_sizes['test']

    print('{}  Acc: {:.4f}'.format('test', acc_val))

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

def consusion_matrix_roc_curve_calc(model, cm_save_name = 'Confusion_Matrix.png', roc_save_name= 'ROC.png'):
    #confusion matrix is built on the test set after training has completed
    print("calculate confusion matrix")
    test_predictions = []
    true_labels = []

    model.eval()
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        for l in labels:
            true_labels.append(l)

        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1)
        for p in test_pred:
            test_predictions.append(p)

    CM = confusion_matrix(true_labels, test_predictions)

    graphs_plotter.plot_confusion_matrix(cm = CM,
                          classes = class_names,
                          normalize = False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues,
                          save_name=cm_save_name)

    #ROC
    fpr, tpr, thresholds = roc_curve(true_labels, test_predictions, pos_label=1)
    auc_score = roc_auc_score(true_labels, test_predictions)
    #plot the roc
    graphs_plotter.roc_plotter(fpr, tpr, auc_score, roc_save_name)

def parameters_statistics_runner(input_model):
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in input_model.parameters())
    # Observe that all parameters are being optimized
    print("total parameters:", total_params)
    total_trainable_params = sum(p.numel() for p in input_model.parameters() if p.requires_grad)
    print("training parameters", total_trainable_params)

def transfer_learner():

    ############ Finetuning the convnet############
    finetuning = True
    if finetuning:
        print("Finetuning the convnet")
        # Load a pretrained model and reset final fully connected layer.
        model_ft = models.resnet18(pretrained=True)
        print("total original parameters:", sum(p.numel() for p in model_ft.parameters()))
        #num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(2048, 2)

        # Find total parameters and trainable parameters
        parameters_statistics_runner(model_ft)

        model_ft = model_ft.to(device)
        # print(model_ft)

        # Set cross entropy loss
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        # Train and evaluate
        best_acc, epochs_array, acc_val, acc_train, loss_val, loss_train, time_elapsed = train_model(model_ft, criterion,
                                                                                                     optimizer_ft,
                                                                                                     exp_lr_scheduler,
                                                                                                     num_epochs=50,
                                                                                                     max_epochs_stop=10,
                                                                                                     model_name='FineTuned_model')
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # Plotting loss and accuracy over epocs
        graphs_plotter.loss_acc_over_epocs_plotter(epochs_array, acc_val, acc_train, metric='Accuracy',
                                                   save_name=os.path.join(root_dir,"FineTuned_NN_Accuracy_over_epochs.png"))
        graphs_plotter.loss_acc_over_epocs_plotter(epochs_array, loss_val, loss_train, metric='Loss',
                                                   save_name=os.path.join(root_dir,"FineTuned_NN_Loss_over_epochs.png"))

        # Test and evaluate
        # model.load_state_dict(torch.load(os.path.join(root_dir, 'models')))
        FineTuned_model = torch.load(os.path.join(root_dir, 'FineTuned_model'))
        # model.load_state_dict(torch.load('mytraining.pt'))

        test_model(FineTuned_model)
        consusion_matrix_roc_curve_calc(FineTuned_model, os.path.join(root_dir,"FineTune_ConfusionMatrix.png"), os.path.join(root_dir,"FineTunedROC.png"))


    ############ConvNet as fixed feature extractor###############
    train_last_layers = True
    if train_last_layers:
        print('ConvNet as fixed feature extractor')
        # Here, we need to freeze all the network except the final layer. We need
        # to set ``requires_grad == False`` to freeze the parameters so that the
        # gradients are not computed in backward().
        #
        # You can read more about this in the documentation
        # here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>
        model_conv = torchvision.models.resnet18(pretrained=True)
        # fix first 4 layers of vgg
        fixed_layers = 0
        for child in model_conv.children():
            fixed_layers += 1
            if fixed_layers < 4:
                for param in child.parameters():
                    param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(2048, 2)

        parameters_statistics_runner(model_conv)

        model_conv = model_conv.to(device)
        # print(model_conv)

        criterion = nn.CrossEntropyLoss()
        # Observe that only parameters of final layer are being optimized as opposed to before.
        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

        # Train and evaluate
        # On CPU this will take about half the time compared to previous scenario.
        # This is expected as gradients don't need to be computed for most of the network
        # Train and evaluate
        best_acc, epochs_array, acc_val, acc_train, loss_val, loss_train, time_elapsed = train_model(model_conv, criterion,
                                                                                                     optimizer_conv,
                                                                                                     exp_lr_scheduler,
                                                                                                     num_epochs=50,
                                                                                                     max_epochs_stop=10,
                                                                                                     model_name='Pretrained_model')
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # Plotting loss and accuracy over epocs
        graphs_plotter.loss_acc_over_epocs_plotter(epochs_array, acc_val, acc_train, metric='Accuracy',
                                                   save_name="Pretrained NN_Accuracy_over_epochs.png")
        graphs_plotter.loss_acc_over_epocs_plotter(epochs_array, loss_val, loss_train, metric='Loss',
                                                   save_name="Pretrained NN_Loss_over_epochs.png")

        # Test and evaluate
        # model.load_state_dict(torch.load(os.path.join(root_dir, 'models')))
        Pretrained_model = torch.load(os.path.join(root_dir, 'Pretrained_model'))
        # model.load_state_dict(torch.load('mytraining.pt'))

        test_model(Pretrained_model)
        consusion_matrix_roc_curve_calc(Pretrained_model, os.path.join(root_dir,"Pretrained_ConfusionMatrix.png"), os.path.join(root_dir,"Pretrained_ROC.png"))

    plt.ioff()


if __name__ == "__main__":
    # Define path to the data directory
    root_dir = os.getcwd()
    data_dir = os.path.join(os.path.join(root_dir, 'data'), 'chest_xray')

    #load datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'val', 'test']}
    global dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in
                   ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    print("dataset_sizes", dataset_sizes)

    class_names = image_datasets['train'].classes
    # classes are Pneumonia/Normal
    print("class_names", class_names)

    run_label_stats = False
    if run_label_stats == True:
        # Count how many in datasets are of each class
        # Observe that datesets are rather balanced
        for c in ['train', 'val', 'test']:
            # number of normal_cases:
            # train: 3875 out of 5216, validation: 8 out of 16, test: 390 out of 624
            print("for",c,"number of normal_cases", label_counter_in_dataset(dataloaders[c]), "out of", dataset_sizes[c])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Visualize a few training images so as to understand the data augmentations
    # Get a batch of training data
    # inputs contains 4 images because batch_size=4 for the dataloaders
    inputs, classes = next(iter(dataloaders['train']))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

    transfer_learner()
