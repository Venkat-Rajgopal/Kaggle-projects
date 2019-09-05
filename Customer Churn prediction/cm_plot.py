import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_cm(cm, classes, mtd, title='Confusion matrix', cmap=plt.cm.Blues):
    print("\nConfusion Matrix\n", cm)
    
    #fig1 = plt.figure(figsize=(5, 5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid()

    # Saving parameter
    #fig1.savefig(mtd+title+'.png', bbox_inches='tight')
    #plt.show()