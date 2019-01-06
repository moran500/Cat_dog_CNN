
# How to build it:
# -   iterate through all the train imagines and also the validation imagine and change it to gray with cv2.imread()
#     this function will transform the picture to array where each number is one pixel of the picture
# -   then we need to set all picture to the same size for example to 50x50 pixels, this will be done with the cv2.resize() function
# -   now we can create the training set, it should be list of imagines which we created with cv2.imread() function, imporant is to remember
#     that we have to also label the training data if it is dog or cat and it has to be number, because the ANN are working with numbers.
#     so we will say that 0 will be cat and 1 will be dog and then we need to shuffle data that we will not have all dog and then all cats
#     this you can do with from lib random and function shuffle
# -   then you have to split the training data to X variable and y variable and of course we have to have numpy arrays with proper shape
#     so np.array("something").reshape("here you have to pass the proper shape in X,Y,Z")
# -   when you have this, it means that you have ready the data (imagines) and you dont want to do it again and again so you should store it
#     this you will do with lib pickle and function dump and it is like writing to the file so you need to open file.pickle and then use 
#     dump function and then close it, this we will do for X variable and also for y variable and when you rerun your model you just read from it
# -   now is the time to create the model, but before that you have to scale the data, it means that you have to transform the data in values 
#     between 0 and 1, for this you can use the Normalize module in Keras. MOdel you should build in the seperated Module where you have to 
#     with pickle first import data which you preprocessed
# -   now define the model, it will contains of Convolution layer, Maxpooling layer and fully connected layer. Convoltion layer and fully 
#     connected layer have to contain the activation function and it will be RELU and dont forget on the Dropout()
# -   before the fully connected layer you have to flatten it, you have to test why.
# -   the last thing is the output layer and then to compile it with some loss function and metrics and optimizer (choose which) google it 

# -   Dont forget on Batch normalization and dropout, in video you saw that the dropout was add only to fully connected layers

#     Plan is: to create this CNN and then use it for building the GAN for this and potencially use mongodb for storing the pictures after 
#     processing

import cv2
import os
import pickle
from random import shuffle

X_train = []
y_train = []
X_test = []
y_test = []

def load_data():
    # this will set up the path to the pictures
    img_dog_train_dir = os.getcwd() + "/img/training_set/dogs/"
    img_cat_train_dir = os.getcwd() + "/img/training_set/cats/"
    img_dog_test_dir = os.getcwd() + "/img/test_set/dogs/"
    img_cat_test_dir = os.getcwd() + "/img/test_set/cats/"
    # this will get from the folder the list of the file names
    img_dog_train_names = os.listdir(img_dog_train_dir)
    img_cat_train_names = os.listdir(img_cat_train_dir)
    img_dog_test_names = os.listdir(img_dog_test_dir)
    img_cat_test_names = os.listdir(img_cat_test_dir)
    

    
    # load Dogs train data
    for img in img_dog_train_names:
        # load picture in Gray scale
        t_img = cv2.imread(img_dog_train_dir + img, cv2.IMREAD_GRAYSCALE)
        t_img = cv2.resize(t_img, (50,50))
        X_train.append(t_img)
        # number 1 will be Dog
        y_train.append(1)
    
    # load Cats train data
    for img in img_cat_train_names:
        # load picture in Gray scale
        t_img = cv2.imread(img_cat_train_dir + img, cv2.IMREAD_GRAYSCALE)
        t_img = cv2.resize(t_img, (50,50))
        X_train.append(t_img)
        # number 1 will be Cat
        y_train.append(0)
    
    # load Dog test data
    for img in img_dog_test_names:
        # load picture in Gray scale
        t_img = cv2.imread(img_dog_test_dir + img, cv2.IMREAD_GRAYSCALE)
        t_img = cv2.resize(t_img, (50,50))
        X_test.append(t_img)
        # number 1 will be Dog
        y_test.append(1)
    
    # load Cat test data
    for img in img_cat_test_names:
        # load picture in Gray scale
        t_img = cv2.imread(img_cat_test_dir + img, cv2.IMREAD_GRAYSCALE)
        t_img = cv2.resize(t_img, (50,50))
        X_test.append(t_img)
        # number 0 will be Cat
        y_test.append(0)

def pickle_data():
    # this will store to folder pickles the data as python objects
    pickle.dump(X_train, open("pickles/X_train.pickle", "wb"))
    pickle.dump(y_train, open("pickles/y_train.pickle", "wb"))
    pickle.dump(X_test, open("pickles/X_test.pickle", "wb"))
    pickle.dump(y_test, open("pickles/y_test.pickle", "wb"))

def unpickle_data():
    # this will load data from pickles folder to the list objects
    X_train = pickle.load(open("pickles/X_train.pickle", "rb"))
    y_train = pickle.load(open("pickles/y_train.pickle", "rb"))
    X_test = pickle.load(open("pickles/X_test.pickle", "rb"))
    y_test = pickle.load(open("pickles/y_test.pickle", "rb"))
    return X_train, y_train, X_test, y_test


def shuffle_data(X_train, y_train, X_test, y_test):
    # this will concatenate X data and y data
    shuffle_train = list(zip(X_train, y_train)) 
    shuffle_test = list(zip(X_test, y_test))
    # this will randomly shuffle 2 lists together
    shuffle(shuffle_train)
    shuffle(shuffle_test)
    # this will split back already shuffled X and y data
    X_train, y_train = zip(*shuffle_train)
    X_test, y_test = zip(*shuffle_test)
    return X_train, y_train, X_test, y_test
    

 
X_train, y_train, X_test, y_test = unpickle_data() 
X_train, y_train, X_test, y_test = shuffle_data(X_train, y_train, X_test, y_test)


print(y_test) 
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))


# show loaded picture
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    