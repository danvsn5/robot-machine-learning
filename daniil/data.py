import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from skimage.transform import resize
from skimage.io import imread

from sklearn import svm
from sklearn.metrics import classification_report


import joblib
import torch as pytorch


print("Method run at: ", pd.to_datetime('today'))

# archive folder from previous directory 


# put each class variable into an array


data_dir = './datasetNEW'


flat_data_arr=[]


target_arr=[]

labels = os.listdir(data_dir)

# runs through all images on dataset flattening each image and resizing it to 32x32
for identification_tag in labels:
    print(identification_tag)
    class_path = os.path.join(data_dir, str(identification_tag))
    for img_name in os.listdir(class_path):
        
        img_path = os.path.join(class_path, img_name)
        
        img_array=imread(os.path.join(class_path,img_name))

        img_resized=resize(img_array,(128,128,3))
        
        flat_data_arr.append(img_resized.flatten())
              
        target_arr.append(identification_tag)
        

print(len(flat_data_arr))

# converts arrays to numpy arrays for better performance
image_data = np.array(flat_data_arr)
target = np.array(target_arr)

df = pd.DataFrame(image_data)
df['target'] = target
print(df)

# split the data into training and testing data

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

print("spliting data")
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=26, stratify=y)


print("Begun at: ", pd.to_datetime('today'))

# runs for loops creating the model with different parameters saving all models to joblib files to be tested in a differnt file

for kernel in ['poly' , 'rbf']:
    for C in [0.1, 1]:
        svc = svm.SVC(kernel=kernel, C=C, max_iter=1000, verbose=0)
            
        svc.fit(x_train,y_train)

        # create classification report for the newly created model
        y_pred = svc.predict(x_test)
        print(classification_report(y_test, y_pred))
        print("report completed at ", pd.to_datetime('today'))

        joblib.dump((svc, x_test, y_test), f'svc_model_{kernel}_{C}.joblib')
        print(f"Model saved as svc_model_{kernel}_{C}.joblib")
        print("saved at ", pd.to_datetime('today'))

for kernel in ['poly' , 'rbf']:
    for C in [0.1, 1]:
        pytorch.save(svc, f'svc_model_{kernel}_{C}.pth')
        print(f"Model saved as svc_model_{kernel}_{C}.pth")
