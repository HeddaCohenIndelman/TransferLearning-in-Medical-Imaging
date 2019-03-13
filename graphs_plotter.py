
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_name='confusion_matrix.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_name)
    plt.clf()
    plt.cla()
    plt.close()

def roc_plotter(fpr, tpr, auc_score, savename):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic on Test set')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(savename)
    plt.clf()
    plt.cla()
    plt.close()

def loss_acc_over_epocs_plotter(epochs_array, m_val, m_train, metric, save_name):
    #mteric is normally accuravy or loss
    fig_a, ax_a = plt.subplots()
    ax_a.plot(epochs_array, m_val, label='Validation' + metric)
    ax_a.plot(epochs_array, m_train, label='Training' + metric)
    ax_a.set(xlabel='epochs', ylabel=metric, title=metric + ' over epochs')
    ax_a.grid()
    ax_a.legend(['train', 'validation'], loc='best')
    fig_a.savefig(save_name)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()