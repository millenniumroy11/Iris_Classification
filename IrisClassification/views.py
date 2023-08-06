from django.shortcuts import render
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def index(request):
    return render(request, "index.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    data = pd.read_csv('E:\Iris_Classification_Project\Iris.csv')

    X = data.drop(['Id', 'Species'], axis=1)
    y = data['Species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    knn = KNeighborsClassifier(n_neighbors=12)
    knn.fit(X, y)

    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])

    pred = logreg.predict(np.array([var1,var2,var3,var4]).reshape(1,-1))
    #pred = round(pred[0])

    price = "We Found It As  : "+str(pred)

    return render(request, "predict.html", {"result2":price})
