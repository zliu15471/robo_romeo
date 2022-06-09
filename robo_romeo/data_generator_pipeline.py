from typing_extensions import Self
import tensorflow as tf
import pandas as pd

class GeneratorPipeline(tf.keras.utils.Sequence):

    def __init__(self, X1, X2, y, batch_size,shuffle=True):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def prepare_training_data(self):


        # data = pd.DataFrame(({'X1': X1, 'X2': X2, 'y': y}, columns=['X1', 'X2', 'y]))?
        pass


    def save_encoded_images(self):
        # split sequences
        # encode images and save
        # create column for encoded image path
        pass

    def save_caption_sequence(self):
        pass

    def save_predict_sequence(self):
        pass

    def __getitem__(self,index):

        self.load_image("../raw_data/Flickr8k_text/.......")

        X1 = self.X1[index * self.batch_size:(index + 1) * self.batch_size]
        X2 = self.X2[index * self.batch_size:(index + 1) * self.batch_size]
        y = self.y[index * self.batch_size:(index + 1) * self.batch_size]

        return [X1,X2], y

    def on_epoch_end(self):
        # if self.shuffle:
            # data.sample(frac=1).reset_index(drop=True)
        pass

    def __len__(self):
        return len(self.X1)//self.batch_size
