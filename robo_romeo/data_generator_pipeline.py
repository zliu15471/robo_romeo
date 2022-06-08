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


        # data = pd.DataFrame(({'X1': X1, 'X2': X2, 'y': y}, columns=['X1', 'X2', 'y]))
        pass
        pass


    def save_encoded_images(self):
        pass

    def __getitem__(self,idx):

        self.load_image("../raw_data/Flickr8k_text/.......")

        X1 = self.X1[idx * self.batch_size:(idx + 1) * self.batch_size]
        X2 = self.X2[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return [X1,X2], y

    def on_epoch_end(self):
        # if self.shuffle:
            # data.sample(frac=1).reset_index(drop=True)
        pass

    def __len__(self):
        return len(self.X1)//self.batch_size
