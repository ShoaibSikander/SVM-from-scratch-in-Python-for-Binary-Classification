import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Scratch Python Implementation!!!")
data = pd.read_csv('/home/shoaib/Github/Support_Vector_Machine_Binary_Classification_from_scratch/Dataset_Iris.csv')

target = data['Species']

rows = list(range(100,150))
data = data.drop(data.index[rows])

sepal_length = data['SepalLengthCm']
petal_length = data['PetalLengthCm']

setosa_sepal_length = sepal_length[:50]
setosa_petal_length = petal_length[:50]

versicolor_sepal_length = sepal_length[50:]
versicolor_petal_length = petal_length[50:]

plt.figure(figsize=(8,6))
plt.scatter(setosa_sepal_length, setosa_petal_length, marker='8', color='m', label='Iris-Setosa')
plt.scatter(versicolor_sepal_length, versicolor_petal_length, marker='D', color='b', label='Iris-Versicolor')
plt.title('Binary Classification of Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.grid()
leg = plt.legend(loc='lower right');
plt.show()

data = data.drop(['Id','SepalWidthCm','PetalWidthCm'],axis=1)

rows = list(range(100,150))
target = target.drop(target.index[rows])

Y = []
for val in target:
    if(val == 'Iris-setosa'):
        Y.append(-1)
    else:
        Y.append(1)
        
data = data.drop(['Species'],axis=1)

X = data.values.tolist()

X, Y = shuffle(X,Y)

x_train = []
y_train = []
x_test = []
y_test = []
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, test_size=0.1)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
y_train = y_train.reshape(90,1)
y_test = y_test.reshape(10,1)

train_feature_1 = x_train[:,0]
train_feature_2 = x_train[:,1]
train_feature_1 = train_feature_1.reshape(90,1)
train_feature_2 = train_feature_2.reshape(90,1)

weights_feature_1 = np.zeros((90,1))
weights_feature_2 = np.zeros((90,1))

epochs = 1
alpha = 0.0001

while(epochs < 10000):
    y = weights_feature_1 * train_feature_1 + weights_feature_2 * train_feature_2
    prod = y * y_train
    #print(epochs)
    count = 0
    for val in prod:
        if(val >= 1):
            cost = 0
            weights_feature_1 = weights_feature_1 - alpha * (2 * 1/epochs * weights_feature_1)
            weights_feature_2 = weights_feature_2 - alpha * (2 * 1/epochs * weights_feature_2)     
        else:
            cost = 1 - val 
            weights_feature_1 = weights_feature_1 + alpha * (train_feature_1[count] * y_train[count] - 2 * 1/epochs * weights_feature_1)
            weights_feature_2 = weights_feature_2 + alpha * (train_feature_2[count] * y_train[count] - 2 * 1/epochs * weights_feature_2)
        count += 1
    epochs += 1
    
index = list(range(10,90))
weights_feature_1 = np.delete(weights_feature_1,index)
weights_feature_2 = np.delete(weights_feature_2,index)

weights_feature_1 = weights_feature_1.reshape(10,1)
weights_feature_2 = weights_feature_2.reshape(10,1)

test_feature_one = x_test[:,0]
test_feature_two = x_test[:,1]

test_feature_one = test_feature_one.reshape(10,1)
test_feature_two = test_feature_two.reshape(10,1)

y_pred = weights_feature_1 * test_feature_one + weights_feature_2 * test_feature_two

predictions = []
for val in y_pred:
    if(val > 1):
        predictions.append(1)
    else:
        predictions.append(-1)
        
print("Accuracy : ", accuracy_score(y_test,predictions))

print("Implementation using classifier from Scikit-Learn")

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='linear')
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print("Accuracy : ", accuracy_score(y_test,y_pred))
