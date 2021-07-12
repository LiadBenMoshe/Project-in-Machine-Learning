from Adaboost import _Adaboost 
from Svm import _Svm 
from knn import _KNN
from NN import _NN


#constuctors for all clisifiers
Portuguese_adaboost = _Adaboost("student-por.csv")
math_adaboost = _Adaboost("student-mat.csv")

Portuguese_svm =_Svm("student-por.csv")
math_svm =_Svm("student-mat.csv")

Portuguese_knn=_KNN("student-por.csv")
math_knn=_KNN("student-mat.csv")

Portuguese_NN = _NN("student-por.csv")
math_NN =_NN("student-mat.csv")


print("\n\n~~~~~Q1~~~~~~~")
print("_math_")
print("ada")
math_adaboost.Q1()
print("\n svm")
math_svm.Q1()
print("\n knn")
math_knn.Q1()
print("\n nureal network")
math_NN.Q1()


print("\n_portuguese_")
print("ada")
Portuguese_adaboost.Q1()
print("\n svm")
Portuguese_svm.Q1()
print("\n knn")
Portuguese_knn.Q1()
print("\n nureal network")
Portuguese_NN.Q1()







print("\n\n~~~~~Q2~~~~~~~")
print("_math_")
print("ada")
math_adaboost.Q2()
print("\n svm")
math_svm.Q2()
print("\n knn")
math_knn.Q2()
print("\n nureal network")
math_NN.Q2()


print("\n_portuguese_")
print("ada")
Portuguese_adaboost.Q2()
print("\n svm")
Portuguese_svm.Q2()
print("\n knn")
Portuguese_knn.Q2()
print("\n nureal network")
Portuguese_NN.Q2()




print("\n\n~~~~~Q3~~~~~~~")
print("_math_")
print("ada")
math_adaboost.Q3()
print("\n svm")
math_svm.Q3()
print("\n knn")
math_knn.Q3()
print("\n nureal network")
math_NN.Q3()


print("\n_portuguese_")
print("ada")
Portuguese_adaboost.Q3()
print("\n svm")
Portuguese_svm.Q3()
print("\n knn")
Portuguese_knn.Q3()
print("\n nureal network")
Portuguese_NN.Q3()




print("\n\n~~~~~Q4~~~~~~~")
print("_math_")
print("ada")
math_adaboost.Q4()
print("\n svm")
math_svm.Q4()
print("\n knn")
math_knn.Q4()
print("\n nureal network")
math_NN.Q4()


print("\n_portuguese_")
print("ada")
Portuguese_adaboost.Q4()
print("\n svm")
Portuguese_svm.Q4()
print("\n knn")
Portuguese_knn.Q4()
print("\n nureal network")
Portuguese_NN.Q4()

