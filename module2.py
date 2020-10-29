"""
ANN part
"""
import os
import pandas as pd
import numpy as np

from sklearn import preprocessing

from keras.layers.core import Dense
from keras.layers import Input
from keras.layers import concatenate

from keras.models import Model


def my_preprocessing(csv_path,scaler1=None,scaler2=None,train_frac= 0.7,type =0):
    """
    :param csv_path: type:str()
    :param scaler1: normalize_other_attributes
    :param scaler2: normalize_time_series
    :return: (other_attributes,time_series,lables,scaler1,scaler2)
    """

    """
    preprocessing
    ------------------------------------------------------------------------------------------------
    """

    if os.path.exists(csv_path):
        df1 = pd.read_csv(csv_path, encoding='utf_8_sig')
    else:
        df= pd.read_csv("training_data.csv",encoding='utf-8')
        df.to_csv("training_data0.csv",encoding="utf_8_sig",index=False)

        df1 = pd.read_csv(csv_path, encoding='utf_8_sig')

    df1 = df1.sample(frac=1)     #打乱数据集顺序

    other_attributes=[]
    time_series=[]
    lables = []

    for indexs in df1.index:
        if type == 0:
            a=list(eval(str(df1.loc[indexs].values.tolist()).strip("[]").strip('"')))
        else:
            a = list(df1.loc[indexs].values.tolist())
        other_attribute = a[2:5]   #str(date)不要了, name 用作标签
        time_serie =a[5:184]        #有些数据缺了一个 只能少一个了
        # print(len(time_serie))

        lable = a[1]
        other_attributes.append(other_attribute)
        time_series.append(time_serie)
        lables.append(lable)
    """
    编辑lable
    """
    for i in range(len(lables)):
        if lables[i] == "佳人新材料":
            lables[i] = 0
        elif lables[i] == "洗毛厂":
            lables[i] = 1
        elif lables[i] == "东泰印染":
            lables[i] = 2
        elif lables[i] == "功兴针织":
            lables[i] = 3
    #-------------------------------------------------------------------------------------------------------------
    """
    normalize other_attribute
    """
    if scaler1 is None:
        scaler1 = preprocessing.StandardScaler().fit(other_attributes)
    other_attributes = scaler1.transform(other_attributes)


    """
    normalize time_series
    """
    if scaler2 is None:
        scaler2 = preprocessing.StandardScaler().fit(time_series)
    time_series =scaler2.transform(time_series)

    """
    split train and test
    """
    length = len(other_attributes)
    train_amount = int(length*train_frac)
    train_other_attributes = other_attributes[0:train_amount]
    test_other_attributes = other_attributes[train_amount:]

    train_time_series = time_series[0:train_amount]
    test_time_series = time_series[train_amount:]

    train_lables = lables[0:train_amount]
    test_lables = lables[train_amount:]

    return (train_other_attributes,test_other_attributes,train_time_series,test_time_series,train_lables,test_lables,scaler1,scaler2)

def test(test_time_series,test_other_attributes,test_lables,model):
    test_loss,test_accuracy= model.evaluate([test_time_series,test_other_attributes],test_lables)

    print("Tested acc:",test_accuracy)
    print("Tested loss",test_loss)


def predict(test_time_series,test_other_attributes,model):
    prediction= model.predict([test_time_series,test_other_attributes])
    print(prediction)   #np.argmax: return the index of the max value

def train(train_time_series, train_other_attributes ,train_lables):
    """
    :param other_attributes:
    :param time_series:
    :param lables:
    :return: model
    """
    """
    timeseries input layers
    """
    # the first branch operates on the first input
    inputA = Input(shape=(179,))  # timeseries  3hours*60=180mins
    x = Dense(45, activation="relu")(inputA)
    x = Dense(3, activation="relu")(x)
    x = Model(inputs=inputA, outputs=x)

    """
    other attributes input layers
    """
    inputB = Input(shape=(3,))  # COD\ time_of_discharge \distance_to_outfall
    y = Dense(3, activation="linear")(inputB)
    y = Model(inputs=inputB, outputs=y)
    """
    combined_part input
    """
    combined = concatenate([x.output, y.output])

    z = Dense(4, activation="softmax")(combined)

    model = Model(inputs=[x.input, y.input], outputs=z)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


    model.fit([train_time_series, train_other_attributes], train_lables, epochs=15, verbose=2)
    return model


train_other_attributes,test_other_attributes,train_time_series,test_time_series,train_lables,test_lables,scaler1,scaler2 = my_preprocessing("training_data0.csv")
train_other_attributes2,test_other_attributes2,train_time_series2,test_time_series2,train_lables2,test_lables2=my_preprocessing("real_data.csv",scaler1=scaler1,scaler2=scaler2,train_frac=0,type=0)[0:6]

if __name__ == '__main__' :

    model = train(train_time_series ,train_other_attributes ,train_lables)
    test(test_time_series,test_other_attributes,test_lables,model)
    model.save("model2.h5")
    # model = keras.models.load_model("model.h5")
    #
    # test(test_time_series2,test_other_attributes2,test_lables2,model)
    #
    # predict([np.zeros(179)],[np.zeros(3)],model)  #[[0.1592547  0.24794893 0.38470873 0.20808755]]