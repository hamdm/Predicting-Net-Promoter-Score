import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import export_graphviz
import graphviz

classifiers = ['KNeighbors','LogisticRegression','LogisticRegression_l1','LinearSVC',
               'DecisionTree','RandomForest','GradientBoosting','SVC']


def classifier(X_train,X_test,y_train,y_test):
    
    training_accuracy = []
    testing_accuracy = []
    
    for classifier in classifiers:
    
        if classifier=='KNeighbors':
        #neighborsToTry = range(1,11)
        #training_accuracy = []
        #test_accuracy = []
        #for neighbors in neighborsToTry:
            #kneighborsclassifier = KNeighborsClassifier(n_neighbors=neighbors)
            #kneighborsclassifier.fit(X,y)
            #trainScore = kneighborsclassifier.score(X,y)
            #training_accuracy.append(trainScore)
            #testScore = kneighborsclassifier.score(X_test,y_test)
            #test_accuracy.append(testScore)
        
            kneighborsclassifier = KNeighborsClassifier()
            score_train=kneighborsclassifier.fit(X_train,y_train).score(X_train,y_train)
            score_test = kneighborsclassifier.score(X_test,y_test)        
            
            training_accuracy.append(score_train)
            testing_accuracy.append(score_test)
        
        
        #plt.plot(neighborsToTry,training_accuracy,label='training_accuracy')
        #plt.plot(neighborsToTry,test_accuracy,label='test_accuracy')
        #plt.legend()
        #plt.show()
        
        elif classifier=='LogisticRegression':
            logisticregression = LogisticRegression()
            logisticregression.fit(X_train,y_train)
            #logisticregression.predict(X_test)
            score_train=logisticregression.score(X_train,y_train)
            score_test = logisticregression.score(X_test,y_test)
            #print('The accuracy of Logistic Regression Model is: {}'.format(score))
            
            training_accuracy.append(score_train)
            testing_accuracy.append(score_test)
            
        #plt.plot(logisticregression.coef_.T,'o',label="C=Default")
        #plt.xticks(range(X_train.shape[1]),features,rotation=90)
        #plt.ylimit(-7,7)
        #plt.legend()
        
        elif classifier=="LogisticRegression_l1":
            logisticregression_l1 = LogisticRegression(penalty='l1')
            logisticregression_l1.fit(X_train,y_train)
            #logisticregression_l1.predict(X_test)
            
            score_train=logisticregression_l1.score(X_train,y_train)
            score_test=logisticregression_l1.score(X_test,y_test)
            
            training_accuracy.append(score_train)
            testing_accuracy.append(score_test)
        
        #plt.plot(logisticregression.coef_.T,'o',label="C=Default")
        #plt.xticks(range(X_train.shape[1]),features,rotation=90)
        #plt.ylimit(-7,7)
        #plt.legend()
        
        elif classifier=='LinearSVC':
            linearsvc=LinearSVC()
            score_train = linearsvc.fit(X_train,y_train).score(X_train,y_train)
            score_test=linearsvc.score(X_test,y_test)
            
            training_accuracy.append(score_train)
            testing_accuracy.append(score_test)
            
        elif classifier=='DecisionTree':
        
            tree = DecisionTreeClassifier(random_state=0)
            score_train = tree.fit(X_train,y_train).score(X_train,y_train)
            score_test = tree.score(X_test,y_test)
    
    
            training_accuracy.append(score_train)
            testing_accuracy.append(score_test)
            
        #tree.feature_importances_
        
        #plt.plot(tree.feature_importances_)
        #plt.xticks(range(X_train.shape[1]),features,rotation=90)
        #plt.ylimit(0,1)
        
        elif classifier=='RandomForest':
            forest = RandomForestClassifier(random_state=0)
            score_train = forest.fit(X_train,y_train).score(X_train,y_train)
            score_test = forest.score(X_test,y_test)
            
            training_accuracy.append(score_train)
            testing_accuracy.append(score_test)
            
        #plt.plot(forest.feature_importances_,'o')
        #plt.xticks(range(X_train.shape[1]),features,rotation=90)
        
        elif classifier=='GradientBoosting':
            gbc = GradientBoostingClassifier(random_state=0)
            score_train=gbc.fit(X_train,y_train).score(X_train,y_train)
            score_test = gbc.score(X_test,y_test)
            
            training_accuracy.append(score_train)
            testing_accuracy.append(score_test)
            
        #plt.plot(gbc.feature_importances_,'o')
        #plt.xticks(range(X_train.shape[1]),features,rotation=90)
        
        elif classifier=='SVC':
            svm = SVC()
            score_train = svm.fit(X_train,y_train).score(X_train,y_train)
            score_test = svm.score(X_test,y_test)
            
            training_accuracy.append(score_train)
            testing_accuracy.append(score_test)
            
        
    return training_accuracy,testing_accuracy,classifiers  
        
        

