"""
Neuron networks 
Authors : Uriel Sachs , Liad Ben Moshe
Python 3.8.6
"""

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import numpy as np
from numpy.core.function_base import linspace
from numpy.lib.function_base import average
import pandas as pd
from sklearn.model_selection import train_test_split

class _NN:

    def __init__(self,csv):
        """
        Adapt the data to the algorithm (convert parameters to intgers)
        """
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
        rounds =50
        sum = 0
        sum2 = 0 

        
        self.data['Dalc'] = np.where(self.data.Dalc <4  ,0 , 1)
        self.data['Walc'] = np.where(self.data.Walc <4  ,0 , 1)
      

        Y1 =self.data["Dalc"]
        Y2 =self.data["Walc"]
        X =  self.data[["sex" , "age" , "famsize" ,"Pstatus" , "studytime" ,"Medu" ,"Fedu"  ,  "failures"  , "schoolsup"   , "famsup"   , "paid" ,
        "activities" , "higher" , "internet"  , "romantic" ,"famrel"     , "freetime" ,  "goout", "health" ,  "absences" , "G3" ]]
        for round in range (rounds):
            #Run adaboost with Dacl
            X_train, X_test, y_train,  y_test = train_test_split(X,Y1, train_size=0.50, random_state=None)
            NN =  MLPClassifier(random_state=1, max_iter=1000,activation='logistic',hidden_layer_sizes=(100,2),early_stopping=True)
            NN.fit(X_train,y_train)
            success = NN.score(X_test,y_test)

            #Run adaboost with Wacl
            X_train, X_test, y_train,  y_test = train_test_split(X,Y2, train_size=0.50, random_state=None)
            NN2 =  MLPClassifier(random_state=1, max_iter=1000,activation='logistic',early_stopping=True)
            NN2.fit(X_train,y_train)
            success2 = NN2.score(X_test,y_test)
            
            sum += success
            sum2 += success2
         
        print ("empirial error of workday alcohol consumption : ", sum/rounds)
        print ("empirial error of  weekend alcohol consumption : ", sum2/rounds)        
        


    def Q2(self,ranging = 3):
        rounds = 50
        sum = 0
        for round in range (rounds):
            X =  self.data[[  "goout", "absences","Dalc" ,"Walc"  ]]    
            Y = self.data["G3"]
            X_train, X_test, y_train,  y_test = train_test_split(X,Y, train_size=0.5, random_state=None)
            NN =  MLPClassifier(random_state=1, max_iter=500,activation='logistic',early_stopping=True)
            NN.fit(X_train,y_train)
            result = NN.predict(X_test)
            y_test= np.array(y_test)
            sucsses = 0 
            for i in range(len(y_test)):
                if result[i] <= y_test[i]+ranging and result[i] >= y_test[i]-ranging :
                    sucsses += 1
            sum += sucsses/len(y_test)
        print("the success rate", sum/rounds)


    def Q3(self ):
        rounds = 50
        sum = 0
        for round in range (rounds):
            X =  self.data[[  "Dalc" ,"Walc" ,"absences","famrel"  ]]    
            Y =self.data["Pstatus"] 
            X_train, X_test, y_train,  y_test = train_test_split(X,Y, train_size=0.50, random_state=None)
            NN =  MLPClassifier(random_state=1, max_iter=500,activation='logistic',early_stopping=True)
            NN.fit(X_train,y_train)
            Accuracy = NN.score(X_test,y_test)
            sum += Accuracy
        print("the success rate of pstatus = ", sum/rounds)



    def Q4(self ):
        #X =  self.data[[  "Dalc" ,"Walc" ,"famrel", "Pstatus", "Mjob" ,"Fjob" ]]    
        X =  self.data[[  "Dalc" ,"Walc" ,"famrel", "Pstatus", "Mjob" ,"Fjob","higher","school" ,"studytime" ,"schoolsup","failures" ]]    

        Y1 =self.data["Medu"]
        Y2 =self.data["Fedu"]
        rounds = 50
        sum = 0
        sum2 = 0
        for round in range (rounds):
            #Run adaboost for predict Mother education
            X_train, X_test, y_train,  y_test = train_test_split(X,Y1, train_size=0.50, random_state=None)
            NN =  MLPClassifier(random_state=1, max_iter=5000,activation='logistic',early_stopping=True,hidden_layer_sizes=(50,4))
            NN.fit(X_train,y_train)
            Accuracy=NN.score(X_test,y_test)
        
            #Run adaboost for predict father education
            X_train, X_test, y_train,  y_test = train_test_split(X,Y2, train_size=0.50, random_state=None)
            NN2 =  MLPClassifier(random_state=1, max_iter=5000,activation='logistic',early_stopping=True,hidden_layer_sizes=(50,4))
            NN2.fit(X_train,y_train)
            Accuracy2=NN2.score(X_test,y_test)
            sum += Accuracy
            sum2 += Accuracy2
        print ("Accuracy of predict mother education : ", sum/rounds)
        print ("Accuracy of predict fathe education : ", sum2/rounds)