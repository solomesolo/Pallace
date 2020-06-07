import pandas as pd
from config import *

class Data(object):
    
    @property
    def train_df(self):
        # read csv
        train_df = pd.read_csv(DATA_PATH+'MURA-v1.1/train_image_paths.csv', header=None, names=['FilePath'])
        # extract Label, BodyPart, StudyType from path of image
        train_df['Label'] = train_df.apply(lambda x:1 if 'positive' in x.FilePath else 0, axis=1)
        train_df['BodyPart'] = train_df.apply(lambda x: x.FilePath.split('/')[2][3:],axis=1)
        train_df['StudyType'] = train_df.apply(lambda x: x.FilePath.split('/')[4][:6],axis=1)
        # add prefix path to dataset to images paths
        train_df['FilePath'] = train_df['FilePath'].apply(lambda x: DATA_PATH+x)
        train_df.set_index(["FilePath", "BodyPart"]).count(level="BodyPart")
        return train_df

    @property
    def valid_df(self):
        valid_df = pd.read_csv(DATA_PATH+'MURA-v1.1/valid_image_paths.csv', header=None, names=['FilePath'])
        # read csv
        valid_df['Label'] = valid_df.apply(lambda x:1 if 'positive' in x.FilePath else 0, axis=1)
        valid_df['BodyPart'] = valid_df.apply(lambda x: x.FilePath.split('/')[2][3:],axis=1)
        valid_df['StudyType'] = valid_df.apply(lambda x: x.FilePath.split('/')[4][:6],axis=1)
        # add prefix path to dataset to images paths
        valid_df['FilePath'] = valid_df['FilePath'].apply(lambda x: DATA_PATH+x)
        return valid_df
    
        
    @property
    def pallace_train_df(self):
        # read csv
        train_df = pd.read_csv(PALLACE_LABELS_PATH)#, header=None)#, names=['FilePath'])
        return train_df

    @property
    def pallace_valid_df(self):
        valid_df = pd.read_csv(PALLACE_LABELS_PATH)#, header=None)#, names=['FilePath'])
        return valid_df
    

    @property
    def train_labels_data(self):
        df = pd.read_csv(DATA_PATH+'MURA-v1.1/train_labeled_studies.csv', names=['FilePath', 'Labels'])
        df['FilePath'] = df['FilePath'].apply(lambda x: DATA_PATH+x)
        return df

    @property
    def valid_labels_data(self):
        df = pd.read_csv(DATA_PATH+'MURA-v1.1/valid_labeled_studies.csv', names=['FilePath', 'Labels'])
        df['FilePath'] = df['FilePath'].apply(lambda x: DATA_PATH+x)
        return df

