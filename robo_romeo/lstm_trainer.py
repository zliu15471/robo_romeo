import numpy as np
import joblib
import pickle
from google.cloud import storage
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.layers import Embedding, LSTM, Add
from tensorflow.keras.utils import to_categorical

### GCP configuration - - - - - - - - - - - - - - - - - - -
GCP_PROJECT_ID = 'robo_romeo'
BUCKET_NAME = 'wagon-data-900-robo_romeo'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -
BUCKET_X1_TRAIN_DATA = 'extracted_features/extract_features_6k.pkl'
BUCKET_X2_TRAIN_DATA = 'processed_captions/cap4'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -
STORAGE_LOCATION = 'models/lstm_model/'
JOBLIB_MODEL = 'trained_lstm_model'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

def get_data():
    # features_file = f"gs://{BUCKET_NAME}/{BUCKET_X1_TRAIN_DATA}"
    features_file = '../extracted_features/extract_features_6k.pkl'
    file = open(features_file, 'rb')
    features = pickle.load(file)
    file.close()

    # captions_file = f"gs://{BUCKET_NAME}/{BUCKET_X2_TRAIN_DATA}"
    captions_file = '../raw_data/captions/cap4'
    file = open(captions_file, 'rb')
    captions = pickle.load(file)
    file.close()

    print("data imported\n")
    return features, captions

def process_data(features, captions):
    sample = 1000

    cap_img_list = captions[0]

    X1 = []
    for cap_img in cap_img_list[:sample]:
        img_feature_matrix = features[cap_img][0]
        X1.append(img_feature_matrix)
    X1 = np.array(X1)

    X2 = np.array(captions[1]).astype(np.uint32)[:sample]

    vocab_size = 7589
    y = np.array([el[0] if len(el)>0 else vocab_size for el in captions[2][:sample]])
    y = to_categorical(y, num_classes=7589)

    print("data processed, ready for training\n")
    print(f"X1 shape:{X1.shape}\nX2 shape:{X2.shape}\ny shape {y.shape}\n")
    print("deleting full dataset from memory, keeping only training data\n")
    features = None
    del features
    captions = None
    del captions
    return X1, X2, y

def train_model(X1, X2, y):

    max_caption_length = 36
    vocab_size = 7589

    inputs2  = Input(shape=(max_caption_length,),name="captions")
    embed_layer = Embedding(vocab_size, 256, mask_zero=True)(inputs2)

    input_encoded = Input(shape=(8,8,1280),name="images_encoded")
    pooling = GlobalAveragePooling2D()(input_encoded)
    cnn_dense = Dense(256, activation='relu')(pooling)

    combine = Add()([embed_layer,cnn_dense])

    lstm_layer = LSTM(256)(combine)
    decoder = Dense(1000, activation='relu')(lstm_layer)
    outputs = Dense(vocab_size, activation='softmax')(decoder)

    model = Model(inputs=[input_encoded, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy' , optimizer='adam',metrics = 'accuracy')

    print("model instantiated\n")

    model.fit(x=(X1, X2), y=y, batch_size=32, epochs=10, verbose=0)
    print("model trained\n")

    return model

def upload_model_to_gcp():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename(JOBLIB_MODEL)

def saving_model(model):
    save_model(model, JOBLIB_MODEL, save_format='h5')
    print("model saved locally\n")
    # upload_model_to_gcp()
    # print("model uploaded to gcp storage\n")

if __name__ == '__main__':
    features, captions = get_data()
    X1_train, X2_train, y_train = process_data(features, captions)
    trained_model = train_model(X1_train, X2_train, y_train)
    saving_model(trained_model)
