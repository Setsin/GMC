
from bigml.api import BigML
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import sklearn as skgitr
import numpy as np
import pandas as pd

api = BigML ('rodolphevdl', 'a47cb24c1d3a04762b4b5db6ece5e4c915efa86c',project = 'project/5d9d9dd05a2139288900006f')
source = api.create_source('KFPFull.csv')
dataset = api.create_dataset(source)

train_dataset = api.create_dataset(dataset, {"name": "KFPFull|train 80% ", "sample_rate": 0.8})
test_dataset = api.create_dataset(dataset, {"name": "KFPFull|validation 20% ", "sample_rate": 0.2})

ensemble_args = {"objective_field": "SeriousDlqin2yrs"}
ensemble = api.create_ensemble(train_dataset, ensemble_args)

from pandas import read_csv


df = read_csv("KFPFull.csv")

prediction_args = {"name": "my prediction"}
batch_prediction = api.create_batch_prediction(ensemble, train_dataset, {"all_fields": True})

api.ok(batch_prediction)

api.download_batch_prediction(batch_prediction, filename='KFPprediction.csv')



