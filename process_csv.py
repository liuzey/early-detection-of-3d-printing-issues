import os
import pandas as pd

origin = pd.read_csv("./train.csv")
train_list = os.listdir("./images/Train/101") + \
             os.listdir("./images/Train/102") + \
             os.listdir("./images/Train/103") + \
             os.listdir("./images/Train/104")
valid_list = os.listdir("./images/Valid/101") + \
             os.listdir("./images/Valid/102") + \
             os.listdir("./images/Valid/103") + \
             os.listdir("./images/Valid/104") + \
             os.listdir("./images/Valid/022")
# print(train_list)
# print(valid_list)

train_list = [int(item) for item in train_list]
valid_list = [int(item) for item in valid_list]

train_csv = origin[origin["print_id"].isin(train_list)]
print(train_csv.shape[0])
train_csv.to_csv("./little_train.csv", index=False)

valid_csv = origin[origin["print_id"].isin(valid_list)]
print(valid_csv.shape[0])
valid_csv.to_csv("./little_valid.csv", index=False)

