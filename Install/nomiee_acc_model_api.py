import numpy as np
import pyodbc
import pandas as pd

from sklearn.preprocessing import Imputer, FunctionTransformer

#How to install xgboost 64bit - https://stackoverflow.com/a/45016496 , !pip install C:\xgboost-0.6-cp36-cp36m-win_amd64.whl


from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)



pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format

# for seaborn issue:
import warnings
warnings.filterwarnings("ignore")

server = 'xxxx'
database = 'LAB'
user = 'ploy'
password = '1234'
con = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + '; UID=' + user + '; PWD=' + password + '')

#######
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import src.scanner as scanner
import os
import csv
import src.util as util
from datetime import datetime
from operator import itemgetter

def normalize_account(account, bin_sizes):
    out = np.zeros((np.sum(bin_sizes)), dtype=np.float32)
    out[0] = util.normalize(account[0], open_stat)
    out[1 + account[1]] = 1
    out[1 + bin_sizes[1]] = util.normalize(account[2], recent_stat)
    out[2 + bin_sizes[1]] = util.normalize(account[3], dormant_stat)
    out[3 + bin_sizes[1] + account[4]] = 1

    return out


def normalize_transactions(trans):
    #print (trans)
    for i in range(len(trans)):
        trans[i][0] = util.normalize(trans[i][0], amount_stat)
        trans[i][1] = util.normalize(trans[i][1], date_stat)

    return trans


def batch_to_one_hot(array_like, bin_sizes):
    #print ("array_like: ", array_like, "-bin_sizes ", bin_sizes, "-\n")
    out = np.zeros((len(array_like), np.sum(bin_sizes)), dtype=np.float32)
    for i in range(len(array_like)):
        last = 0
        for j in range(len(bin_sizes)):
            next_ = (last + bin_sizes[j])
            out[i, last:next_] = util.to_one_hot(array_like[i][j], bin_sizes[j])
            last = next_
    return out


def prep_data(tuples):
    #profiles = []
    trans = []
    labels = []

    for tuple in tuples:
        #profiles.append(normalize_account(tuple[0], [1, len(customer_types), 1, 1, len(units)]))
        trans.append(batch_to_one_hot(normalize_transactions(tuple[0]), [1, 1, len(operation_types)]))
        labels.append(1.0 if tuple[1] else 0)

    return (np.stack(trans), np.stack(labels))

def sort_transaction(transactions):
    return sorted(transactions, key=itemgetter(1))


def sequence_summarize(sorted_transactions):
    out = []
    temp = [0, 0, 0]
    for t in sorted_transactions:
        if t[1] == temp[1] and t[2] == temp[2]:
            temp[0] = temp[0] + t[0]
        else:
            out.append(temp)
            temp = [t[0], t[1], t[2]]
    out.append(temp)
    return out[1:]

stime = datetime.utcfromtimestamp(0)
dir_path = os.getcwd()

customer_types = []
units = []
operation_types = []


open_stat = util.get_start_stat()
recent_stat = util.get_start_stat()
dormant_stat = util.get_start_stat()
# freq_stat = util.get_start_stat()
# interest_stat = util.get_start_stat()

amount_stat = util.get_start_stat()
date_stat = util.get_start_stat()

from sklearn.preprocessing import LabelEncoder
from numpy import array
def lblEn (x):
    data = x
    values = array(data)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #print(label_encoder.classes_)
    #print(label_encoder.inverse_transform(label_encoder.classes_))
    #print ("values: ", values)
    #print ("encode: ", integer_encoded)
    return integer_encoded 

from sklearn.preprocessing import StandardScaler
def stdScal (x):
    x = StandardScaler().fit_transform(x.reshape(-1, 1))
    return x


from keras.preprocessing.sequence import pad_sequences
def seq_pad(x):
    padded = pad_sequences(x, dtype=float, padding='pre', maxlen=20)
    return padded

####### Transform Data ######
def transformData(InquiryLogID):
    sql = " SELECT * FROM InquiryLogStatement where InquiryLogID = '" + str(InquiryLogID) + "'"
    df_main2 = pd.read_sql(sql, con)
    df_main2['TransactionDate'] = pd.to_datetime(df_main2['TransactionDate'], format='%d/%m/%Y %H:%M:%S')
    temp = pd.DataFrame({
    'InquiryLogID': df_main2['InquiryLogID'],
    'TransactionDate': df_main2['TransactionDate'],
    'Withdrawal': df_main2['Withdrawal'],
    'Deposit': df_main2['Deposit'],
    })

    temp['txn_type'] = temp['InquiryLogID']
    temp.fillna(0, inplace=True)
    temp['txn_type'] = temp['Withdrawal'].apply(lambda x: "CR" if x == 0 else "DR")

    temp['Amount'] = temp['Deposit']
    temp['Amount'] = None
    temp['Amount'] = temp.apply(lambda x: x['Deposit'] if x['Withdrawal'] == 0 else x['Withdrawal'], axis=1)
    temp['Amount'] = abs(temp['Amount'].apply(lambda x: str(x).replace(',', '')).astype(float))


    df_tran = temp = pd.DataFrame({
            'account_no': temp['InquiryLogID'],
            'from_to_account_no': 0,
            'txn_amount': temp['Amount'],
            'txn_dt': temp['TransactionDate'],
            'txn_hour': 0,
            'txn_type': temp['txn_type'],
            })
    
    a = df_tran['txn_dt']
    a = a.apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    
    from datetime import datetime
    stime = datetime.utcfromtimestamp(0)
    df_tran['txn_dt'] = a.apply(lambda y: (datetime.strptime(y, '%Y-%m-%d %H:%M:%S') - stime).total_seconds())

    df_tran['txn_type'] = lblEn(df_tran['txn_type'])
    
    df_tran['txn_amount'] = stdScal(df_tran['txn_amount'])
    df_tran['txn_dt'] = stdScal(df_tran['txn_dt'])
    #print(df_tran)
    account_transactions = {}
    #print("reading transaction file...")
    arr_main = []
    for index, row in df_tran.iterrows():
        transaction = []
        # transaction.append(row[0])  # account id
        transaction.append(util.collect_statistics(amount_stat, row[2]))   #amount
        transaction.append(util.collect_statistics(date_stat,(row[3])))
        transaction.append(util.check_and_update_list(operation_types, row[5]))  # type
        
        if row[0] in account_transactions:
            account_transactions[row[0]].append(transaction)
        else:
            account_transactions[row[0]] = [transaction]

    #print(account_transactions, "\n")
    transactions = sequence_summarize(sort_transaction(account_transactions[str(InquiryLogID)]))
    arr_main.append(transactions)
    #print("----------\n", arr_main, "\n--------")
    #print("**")
        
   
    dataset = seq_pad(arr_main)
    #dataset.shape
    return dataset


#################### API ########################

# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

class Noms(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id',type=int)
        args = parser.parse_args()
        
        id = args['id']
        data = searchById(id)

        # load json and create model
        json_file = open('nominee_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        
        # load weights into new model
        loaded_model.load_weights("nominee_model.h5")
        predict = loaded_model.predict_classes(data)
        return str(predict)

        
 
def searchById(id):
    id = int(id)
    dataset = transformData(id)
    print(dataset)
    return dataset
         
         

api.add_resource(Noms, '/noms')



##
## Actually setup the Api resource routing here
##



if __name__ == '__main__':
    app.run(debug=True)