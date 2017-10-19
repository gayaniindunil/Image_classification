import keras
from keras.datasets import cifar10
from matplotlib import pyplot

def model_input(train_dir = None , test_dir = None ):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # create a grid of 3x3 images
    # for i in range(0, 9):
    # 	pyplot.subplot(330 + 1 + i)
    # 	pyplot.imshow(x_train[i])
    # # show the plot
    # pyplot.show()

    return (x_train, y_train), (x_test, y_test)
