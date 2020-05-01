from keras.applications.vgg19 import VGG19,preprocess_input
from keras import layers
from keras.optimizers import Adam,SGD,RMSprop
from keras.utils import to_categorical
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imutils import paths
import random
import cv2
import os
import keras
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn import metrics
from keras.callbacks import EarlyStopping,ModelCheckpoint,CSVLogger
from keras.models import Model
from keras import regularizers
from sklearn.utils import class_weight
import tensorflow as tf
import tensorflow
import numpy
from keras.models import load_model
import hickle as hkl
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys
import pandas as pd
import seaborn as sns

physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

num_epochs = 1
init_lr = 5e-3
n_classes = 19
batch_size = 10
cvscores = []

model_name = "VGG19"
dataset_name = "CEBDB"


def poly_decay(epoch):

    maxEpochs = num_epochs
    baseLR = init_lr
    power = 1.0
    
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    return alpha

def load_custom_model():
    
    baseModel = VGG19(include_top=False,weights=None,input_shape=(224,224,3))
    baseModel.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    model = keras.Sequential()
    model.add(baseModel)
    model.add(layers.GlobalAveragePooling2D(name="gmp"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512,activation="relu",kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.Dense(512,activation="relu",kernel_regularizer=regularizers.l2(0.0005)))
    model.add(layers.Dense(n_classes,activation="softmax"))
    
    
    print("loaded weights")
    
    return model

def load_data():
    
    print("[INFO] loading {} data...".format(dataset_name))
    data = []
    labels = []
    print("Loading images")
    dataset = "users{}/".format(dataset_name)
    imagePaths = sorted(list(paths.list_images(dataset)))
    random.seed(42)
    random.shuffle(imagePaths)
    
    for imagePath in imagePaths:
        
        image  = cv2.imread(imagePath)
        image = img_to_array(image)
        image = preprocess_input(image,mode="tf")
        data.append(image)
        labelz = imagePath.split(os.path.sep)[-2] 
        label = labelz[5:]
        labels.append(label)
    
    print("finished loading images")
    data = np.array(data, dtype="float32") 
    print(data.shape)
    labels = np.array(labels)
    #class_weights = class_weight.compute_class_weight('balanced',np.unique(labels),labels)
    (trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.20,shuffle=True,stratify=labels,random_state=42)
    
    return trainX, testX, trainY, testY
    
def train_model(trainX,trainY):
        
        
     for train, test in kfold.split(trainX,trainY):
        
        print("Training images = ",len(train))
        print("Testing images = ",len(test))
 
        class_weights = class_weight.compute_class_weight('balanced',np.unique(trainY[train]),trainY[train])
        opt = SGD(lr = init_lr, momentum = 0.9)
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=15,restore_best_weights=True)
        mc = ModelCheckpoint('data/best_model_{}_{}.h5'.format(model_name,dataset_name), monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
        csv_logger = CSVLogger("{}_{}_log.csv".format(model_name,dataset_name), append=True)
        callbacks = [es, mc,csv_logger]
        
        model = load_custom_model()
        
        model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
        print("[INFO] training network...")

        hist = model.fit(trainX[train],to_categorical(trainY[train], num_classes=n_classes), 
                         batch_size=batch_size,
                         callbacks=callbacks,
                         validation_data=(trainX[test],to_categorical(trainY[test],num_classes=n_classes)),
                         class_weight=class_weights,
                         epochs=num_epochs, verbose=1,shuffle=True)
        
        scores = model.evaluate(trainX[test],to_categorical(trainY[test], num_classes=n_classes),verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
        cvscores.append(scores[1] * 100)
        
        break
    
     print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
     print(cvscores)
    
     return hist

def draw_plot(hist):
    
    plt.figure()
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("plots/accuracy_{}_{}.eps".format(model_name,dataset_name),format='eps')
    
    plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("plots/loss_{}_{}.eps".format(model_name,dataset_name),format='eps')
    plt.show()
    
def load_bestModel():
        
    bestModel = load_model('data/best_model_{}_{}.h5'.format(model_name,dataset_name))
    _,score = bestModel.evaluate(testX,testY,verbose=1)
    print(score*100)
    
    return bestModel

def accept_reject(my_label,y_pred):
    
    f = open("data/accept_reject_{}_{}.txt".format(model_name,dataset_name),"w")
    
    
    for i in range(0,len(my_label)):
        if my_label[i] == y_pred[i]:
            f.write(str("Label - ") + str(my_label[i]) + str("   Predicted -  ") + str(y_pred[i]) + str("   Accept") + '\n') 
        else:
            f.write(str("Label - ") + str(my_label[i]) + str("   Predicted -  ") + str(y_pred[i]) + str("   Reject") + '\n') 
        
    f.close()
    
def save_prediction_data(bestModel):
    
    y_pred = bestModel.predict(testX)
    y_pred = np.argmax(y_pred,axis=1)
    y_pred2 = bestModel.predict(testX)
    
    my_label = np.argmax(testY,axis=1)
    accept_reject(my_label,y_pred)
    
    data = {'testY': testY, 'y_pred2': y_pred2}
    hkl.dump(data,'data/{0}_{1}.hkl'.format(model_name,dataset_name))
    
    return y_pred,y_pred2

def gen_confusion_matrix(testY,y_pred):
    
    results = confusion_matrix(np.argmax(testY,axis=1),y_pred)
    classes=[]
    for i in range(0,n_classes):
        classes.append(i)
        
    TN = []
    for i in range(n_classes):
         temp = np.delete(results, i, 0)   # delete ith row
         temp = np.delete(temp, i, 1)  # delete ith column
         TN.append(sum(sum(temp)))
    
    FP = results.sum(axis=0) - np.diag(results)  
    FN = results.sum(axis=1) - np.diag(results)  
    TP = np.diag(results)    
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    EER = (FP+FN)/(TP+TN+FP+FN)
    
    return FPR,FNR,EER
    
def plot_roc_eer(FPR,FNR,EER):
    
    plt.figure()
    EER = np.mean(EER)*100
    print(" EER = ",EER)
    plt.plot(np.sort(FPR)) # Sorted in ascending order
    plt.plot(np.sort(FNR)[::-1]) # Sorted in descending order
    plt.savefig("./plots/eer_{}_{}.eps".format(model_name,dataset_name),format='eps')
    false_negative_rate = sum(FNR)*100/n_classes
    false_positive_rate = sum(FPR)*100/n_classes
    print("False negative rate = ",false_negative_rate)
    print("False positive rate = ",false_positive_rate)
    f = open("data/metrics.txt_{}_{}".format(model_name,dataset_name),"w")
    f.write(str(false_negative_rate)+"%"+'\n'+str(false_positive_rate)+"%"+'\n'+str(EER)+"%")
    f.close()

def gen_classification_report(testY,y_pred):
    
    distY = np.argmax(testY,axis=1)
    sns.distplot(distY)
    
    plt.figure()
    plt.hist(distY,bins=n_classes)
    plt.savefig("./plots/data_dist_{}_{}.eps".format(model_name,dataset_name),format='eps')

    print(metrics.classification_report(distY, y_pred, digits=3))
    report = metrics.classification_report(distY, y_pred, digits=3,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv("data/metrics2_{}_{}.csv".format(model_name,dataset_name))


trainX, testX, trainY, testY = load_data()

testY = to_categorical(testY, num_classes=n_classes)

kfold = StratifiedKFold(n_splits=5, shuffle=True)

hist = train_model(trainX,trainY)
    
draw_plot(hist)
bestModel = load_bestModel()
y_pred,y_pred2 = save_prediction_data(bestModel) 
FPR,FNR,EER = gen_confusion_matrix(testY,y_pred)    

plot_roc_eer(FPR,FNR,EER)

gen_classification_report(testY,y_pred)

print("----------------DONE------------------")