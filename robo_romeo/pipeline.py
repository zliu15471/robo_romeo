import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from IPython import display
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tqdm.notebook import tqdm
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import os



# function to load documents into memory
def load_doc(filename):
    # Open file to read
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# create dataframe from cleaned descriptions.txt file. pass file path to function to create
def get_dataframe(folder_path):
    dataframe = load_doc(folder_path)
    dataframe = dataframe.split('\n')

    l_all =[]
    for n in dataframe:
        line = {"id": n.split(" ")[0],
            'value': "startsequence " + " ".join(n.split(" ")[1:]) + " endsequence"}
        l_all.append(line)

    df_all = pd.DataFrame(l_all)

    t = Tokenizer()
    t.fit_on_texts(df_all.value)

    df_all["value_tokenized"] = t.texts_to_sequences(df_all.value)

    train_ids = np.unique(df_all.id)[:int(0.8*len(np.unique(df_all.id)))]
    test_ids = np.unique(df_all.id)[int(0.8*len(np.unique(df_all.id))):]

    df_train = df_all[df_all.id.isin(train_ids)]
    df_test = df_all[df_all.id.isin(test_ids)]

    return df_train,df_test


class DataPipeline:

    def __init__(self,df,batch_size, vocab_size, img_folder_path,model):
        self.df = df
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.img_folder_path = img_folder_path
        self.prepare_dataset()
        self.encoder_model = model

    def prepare_dataset(self):


        # for loop to append X1,X2,y
        X1,X2,y = [],[],[]

        for idx, data in self.df.iterrows():

            seq = data["value_tokenized"]
            for i in range(1,len(seq)):
                X1.append(data["id"])
                X2.append(seq[0:i])
                y.append(seq[i])

        self.X1,self.X2,self.y = X1,X2,y

    def encode_all_images(self):



        l_toencode = np.unique(self.X1)
        dic_encoded={}
        for image_name in tqdm(l_toencode):
            img_path = self.img_folder_path+image_name + ".jpg"
            arr_path = self.img_folder_path+image_name + ".npy"
            if os.path.exist(img_path):
                img = image.load_img(img_path, target_size=(256,256,3))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                arr= self.encoder_model.predict(x)[0]
                np.save(open(arr_path, 'wb'),arr)

    def load_images_encoded(self, imgs_to_load):


        l_toencode = np.unique(imgs_to_load)
        dic_encoded={}
        for image_name in l_toencode:
            arr_path = self.img_folder_path+image_name + ".npy"
            dic_encoded[image_name] = np.load(open(arr_path, 'rb'))

        features = []
        for image_name in imgs_to_load:
            features.append(dic_encoded[image_name])


        final_array = np.array(features)
        return final_array


    def seq_to_padded(self,seq_to_pad):
        inputs_seq_model = pad_sequences(seq_to_pad,padding='post',maxlen=36)

        return inputs_seq_model


    def to_cat(self, y_to_cat):
        # function to categorical
        y = tf.keras.utils.to_categorical(y_to_cat, num_classes=self.vocab_size+2)
        return y

    def __getitem__(self,idx):


        imgs_to_load = self.X1[idx * self.batch_size : (idx +1) * self.batch_size]
        x1_batch = self.load_images_encoded(imgs_to_load)

        seq_to_pad = self.X2[idx * self.batch_size : (idx +1) * self.batch_size]
        x2_batch = self.seq_to_padded(seq_to_pad)

        y_to_cat = self.y[idx * self.batch_size : (idx +1) * self.batch_size]
        y_batch = self.to_cat(y_to_cat)

        return ([x1_batch,
                x2_batch],
                y_batch)

    def __len__(self):
        return len(self.X1)// self.batch_size
