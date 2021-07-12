"""
svm 
Authors : Uriel Sachs , Liad Ben Moshe
Python 3.8.6
"""

from sklearn import svm
import numpy as np
from numpy.core.function_base import linspace
from numpy.lib.function_base import average
import pandas as pd
from sklearn.model_selection import train_test_split


class _Svm:

    def __init__(self,csv):
        self.data = pd.read_csv(csv)
        self.mapping = { 'yes' : "1" , 'no' : 0}
        self.data = self.data.replace( to_replace=['yes','no'],value=[1,0])
        self.data.sex = pd.factorize(self.data.sex)[0]
        self.data.famsize = pd.factorize(self.data.famsize)[0]
        self.data.Pstatus = pd.factorize(self.data.Pstatus)[0]
        self.data.Mjob = pd.factorize(self.data.Mjob)[0]
        self.data.Fjob = pd.factorize(self.data.Fjob)[0]
        self.data.school = pd.factorize(self.data.school)[0]


    def Q1(self):
        """
        We adapted the data to the algorithm,To get a logical result.
        we changed the section of alcohol consumption from 1-5 to high or low consumption.
        (4-5 high  consumption , 1-3 not high )
        Dalc - workday alcohol consumption 
        Walc - weekend alcohol consumption 
        """
        self.data['Dalc'] = np.where(self.data.Dalc <4  ,0 , 1)
        self.data['Walc'] = np.where(self.data.Walc <4  ,0 , 1)
        Y1 =self.data["Dalc"]
        Y2 =self.data["Walc"]
        X =  self.data[["sex" , "age" , "famsize" ,"Pstatus" , "studytime" ,"Medu" ,"Fedu"  ,  "failures"  , "schoolsup"   , "famsup"   , "paid" ,
        "activities" , "higher" , "internet"  , "romantic" ,"famrel"     , "freetime" ,  "goout", "health" ,  "absences" , "G3" ]]
        rounds = 60
        sum = 0
        sum2 = 0 
        for round in range (rounds):
        #Run svm to predict workday alcohol consumption
            X_train, X_test, y_train,  y_test = train_test_split(X,Y1, train_size=0.50, random_state=None)
            clf =svm.SVC(kernel="linear")
            clf.fit(X_train,y_train)
            Accuracy = clf.score(X_test,y_test)

        #Run svm to predict workday alcohol consumption
            X_train, X_test, y_train,  y_test = train_test_split(X,Y2, train_size=0.50, random_state=None)
            clf2 =svm.SVC(kernel="linear")
            clf2.fit(X_train,y_train)
            Accuracy2 = clf2.score(X_test,y_test)
            sum += Accuracy
            sum2 += Accuracy2
        print ("Accuracy of workday alcohol consumption : ", sum/rounds)
        print ("Accuracy of  weekend alcohol consumption : ", sum2/rounds)


    def Q2(self,ranging = 3):
        rounds = 20
        sum = 0
        for round in range (rounds):
            X =  self.data[[  "goout", "absences","Dalc" ,"Walc"  ]]    
            Y = self.data["G3"]
            X_train, X_test, y_train,  y_test = train_test_split(X,Y, train_size=0.5, random_state=None)
            clf =svm.SVC(kernel="linear")
            clf.fit(X_train,y_train)
            result = clf.predict(X_test)
            y_test= np.array(y_test)
            #print ("empirial error grade: ", err)
            sucsses = 0 
            for i in range(len(y_test)):
                if result[i] <= y_test[i]+ranging and result[i] >= y_test[i]-ranging :
                    sucsses += 1
            sum += sucsses/len(y_test)
        print("the success rate", sum/rounds)

    
    def Q3(self):
        rounds = 50
        sum = 0
       
        for round in range (rounds):
            X =  self.data[[  "Dalc" ,"Walc" ,"absences","famrel"  ]]    
            Y =self.data["Pstatus"] 
            X_train, X_test, y_train,  y_test = train_test_split(X,Y, train_size=0.50, random_state=None)
            clf =svm.SVC(kernel="linear")
            clf.fit(X_train,y_train)
            Accuracy = clf.score(X_test,y_test)
            sum += Accuracy
        print("the success rate of pstatus = ", sum/rounds)

    def Q4(self,ranging = 3):
       # X =  self.data[[  "Dalc" ,"Walc" ,"famrel", "Pstatus", "Mjob" ,"Fjob" ]]    
        X =  self.data[[  "Dalc" ,"Walc" ,"famrel", "Pstatus", "Mjob" ,"Fjob","higher","school" ,"studytime" ,"schoolsup" ]]    
        Y1 =self.data["Medu"]
        Y2 =self.data["Fedu"]
        rounds = 50
        sum = 0
        sum2 = 0
        for round in range (rounds):
            #Run adaboost for predict Mother education
            X_train, X_test, y_train,  y_test = train_test_split(X,Y1, train_size=0.50, random_state=None)
            clf =svm.SVC(kernel="linear")
            clf.fit(X_train,y_train)
            Accuracy=clf.score(X_test,y_test)
        
            #Run adaboost for predict father education
            X_train, X_test, y_train,  y_test = train_test_split(X,Y2, train_size=0.50, random_state=None)
            clf2 =svm.SVC(kernel="linear")
            clf2.fit(X_train,y_train)
            Accuracy2=clf2.score(X_test,y_test)
            sum += Accuracy
            sum2 += Accuracy2
        print ("Accuracy of predict mother education : ", sum/rounds)
        print ("Accuracy of predict fathe education : ", sum2/rounds)
    