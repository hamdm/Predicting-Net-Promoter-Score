from sklearn.model_selection import KFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
import time



kfold = KFold(n_splits=5,shuffle=True,random_state=0)
SCALER = [None,StandardScaler()]
FEATURES = [40,32,28,26,24]
REDUCER__N_COMPONENTS = [2,6,10,14,18,22]

training_accuracy_gridSearch = []
testing_accuracy_gridSearch = []
training_time_gridSearch = []
prediction_time_gridSearch = []
best_params_classifiers = {}

classifiers = ['KNeighbors','LogisticRegression','LinearSVC',
               'DecisionTree','RandomForest','GradientBoosting','SVC','GaussianNB']

def gridSearch(X_train,X_test,y_train,y_test):


    training_accuracy_gridSearch= []
    testing_accuracy_gridSearch = []
    training_time = []
    prediction_time = []
    best_params_classifiers = {}
    
    for classifier in classifiers:
        if classifier=='KNeighbors':
            
            NEIGHBORS = range(1,11)
            pipe = Pipeline([('scaler',StandardScaler()),
                             ('classifier',KNeighborsClassifier())])
                        
            param_grid = { 'scaler':SCALER,'classifier__n_neighbors': NEIGHBORS}
                        
            grid = GridSearchCV(pipe,param_grid,cv=kfold)
                        
            fit_time = time.time()            
            grid.fit(X_train,y_train)
            f_time = round(time.time()-fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = grid.score(X_train,y_train)
            print('Training Score {}'.format(score_train))
            score_test = grid.score(X_test,y_test)
            print('Testing Score {}'.format(score_test))
            training_accuracy_gridSearch.append(score_train)
            testing_accuracy_gridSearch.append(score_test)
            training_time_gridSearch.append(f_time)
            prediction_time_gridSearch.append(p_time)
                        
            best_params_classifiers['KNeighbors'] = grid.best_params_
            print('Best Parameters {}'.format(grid.best_params_))
            
        elif classifier=='LogisticRegression':
            PENALTY = ['l2','l1']
            C_PARAM_LOG = [0.001,0.01,0.1,1,10,100]
                        
            pipe = Pipeline([
                                ('scaler',StandardScaler()),
                                ('classifier',LogisticRegression())
                                ])
                        
            param_grid = { 'scaler':SCALER,
                                'classifier__penalty':PENALTY,
                                'classifier__C': C_PARAM_LOG,
                                }
            
            grid = GridSearchCV(pipe,param_grid,cv=kfold)
                        
            fit_time = time.time()            
            grid.fit(X_train,y_train)
            f_time = round(time.time()-fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = grid.score(X_train,y_train)
            print('Training Score {}'.format(score_train))
            score_test = grid.score(X_test,y_test)
            print('Testing Score {}'.format(score_test))
            training_accuracy_gridSearch.append(score_train)
            testing_accuracy_gridSearch.append(score_test)
            training_time_gridSearch.append(f_time)
            prediction_time_gridSearch.append(p_time)
                        
            best_params_classifiers['LogisticRegression'] = grid.best_params_
            print('Best Parameters {}'.format(grid.best_params_))
            
        elif classifier == 'LinearSVC':
            C_PARAM_LSVC = [0.001,0.01,0.1,1,10,100]
            
            pipe = Pipeline([
                                ('scaler',StandardScaler()),
                                ('classifier',LinearSVC())
                                ])
                        
            param_grid = {
                                'scaler':SCALER,
                                'classifier__C': C_PARAM_LSVC,
                                }
            
            grid = GridSearchCV(pipe,param_grid,cv=kfold)
                        
            fit_time = time.time()            
            grid.fit(X_train,y_train)
            f_time = round(time.time()-fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = grid.score(X_train,y_train)
            print('Training Score {}'.format(score_train))
            score_test = grid.score(X_test,y_test)
            print('Testing Score {}'.format(score_test))
            training_accuracy_gridSearch.append(score_train)
            testing_accuracy_gridSearch.append(score_test)
            training_time_gridSearch.append(f_time)
            prediction_time_gridSearch.append(p_time)
                        
            best_params_classifiers['LinearSVC'] = grid.best_params_
            print('Best Parameters {}'.format(grid.best_params_))
            
        elif classifier=='DecisionTree':
            CRITERION = ['gini','entropy']
            SPLITTER = ['best','random']
            MIN_SAMPLES_SPLIT = [8,10,12,14,16]
            MAX_DEPTH = [6,12,18]
            CLASS_WEIGHT = ['balanced',None]
                        
            pipe = Pipeline([
                                ('scaler',StandardScaler()),
                                ('classifier',DecisionTreeClassifier())
                                ])
                        
            param_grid = {
                            'scaler': SCALER,
                            'classifier__criterion': CRITERION,
                            'classifier__splitter': SPLITTER,
                            
                            'classifier__min_samples_split': MIN_SAMPLES_SPLIT,
                            'classifier__class_weight': CLASS_WEIGHT,
                        }
            
            grid = GridSearchCV(pipe,param_grid,cv=kfold)
                        
            fit_time = time.time()            
            grid.fit(X_train,y_train)
            f_time = round(time.time()-fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = grid.score(X_train,y_train)
            print('Training Score {}'.format(score_train))
            score_test = grid.score(X_test,y_test)
            print('Testing Score {}'.format(score_test))
            training_accuracy_gridSearch.append(score_train)
            testing_accuracy_gridSearch.append(score_test)
            training_time_gridSearch.append(f_time)
            prediction_time_gridSearch.append(p_time)
                        
            best_params_classifiers['DecisionTreeClassifier'] = grid.best_params_
            print('Best Parameters {}'.format(grid.best_params_))
        
        elif classifier=='RandomForest':
            ESTIMATOR = [10,30,50,70,90,110]
            CRITERION_RFC = ['gini','entropy']
            FEATURES = ['auto',8,10,12]
            MIN_SAMPLES_SPLIT = [2,4,6,8]
            CLASS_WEIGHT = ['balanced',None]
                        
            
            
            pipe = Pipeline([
                                ('scaler',StandardScaler()),
                               
                                ('classifier',RandomForestClassifier(n_jobs=-1,random_state=42))
                                ])
                        
            param_grid = {
                            'scaler': SCALER,
                           
                            'classifier__n_estimators' : ESTIMATOR,
                            'classifier__criterion': CRITERION_RFC,
                            'classifier__max_features': FEATURES,
                            'classifier__min_samples_split': MIN_SAMPLES_SPLIT,
                            'classifier__class_weight': CLASS_WEIGHT,
                        }
            
            grid = GridSearchCV(pipe,param_grid,cv=kfold)
                        
            fit_time = time.time()            
            grid.fit(X_train,y_train)
            f_time = round(time.time()-fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = grid.score(X_train,y_train)
            print('Training Score {}'.format(score_train))
            score_test = grid.score(X_test,y_test)
            print('Testing Score {}'.format(score_test))
            training_accuracy_gridSearch.append(score_train)
            testing_accuracy_gridSearch.append(score_test)
            training_time_gridSearch.append(f_time)
            prediction_time_gridSearch.append(p_time)
                        
            best_params_classifiers['RandomForestClassifier'] = grid.best_params_
            print('Best Parameters {}'.format(grid.best_params_))
            
        elif classifier=='GradientBoosting':
            ESTIMATOR = [10,50,70,100]
            MIN_SAMPLES_SPLIT = [2,4,6,8]
            LEARNINGRATE = [0.001,0.01,0.1]
            MAXDEPTH = [2,3,5,7,9]
            CLASS_WEIGHT = ['balanced',None]
            
            
            pipe = Pipeline([
                                ('scaler',StandardScaler()),
                                
                                ('classifier',GradientBoostingClassifier(random_state=42))
                                ])
                        
            param_grid = {
                            'scaler': SCALER,
                            
                            'classifier__learning_rate': LEARNINGRATE,
                            'classifier__n_estimators' : ESTIMATOR,
                            'classifier__min_samples_split': MIN_SAMPLES_SPLIT,
                            'classifier__max_depth': MAXDEPTH
            }
            
            grid = GridSearchCV(pipe,param_grid,cv=kfold)
                        
            fit_time = time.time()            
            grid.fit(X_train,y_train)
            f_time = round(time.time()-fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = grid.score(X_train,y_train)
            print('Training Score {}'.format(score_train))
            score_test = grid.score(X_test,y_test)
            print('Testing Score {}'.format(score_test))
            training_accuracy_gridSearch.append(score_train)
            testing_accuracy_gridSearch.append(score_test)
            training_time_gridSearch.append(f_time)
            prediction_time_gridSearch.append(p_time)
                        
            best_params_classifiers['GradientBoostingClassifier'] = grid.best_params_
            print('Best Parameters {}'.format(grid.best_params_))
            
        elif classifier=='SVC':
            C_PARAM = [0.001,0.01,0.1,1,10,100]
            GAMMA_PARAM = [0.001,0.01,0.1,1,10,100]
            KERNEL = ['rbf','sigmoid']
                        
            pipe = Pipeline([
                                ('scaler',StandardScaler()),
                               
                                ('classifier',SVC())
                                ])
                        
            param_grid = {
                                'scaler':SCALER,
                                
                                'classifier__C': C_PARAM,
                                'classifier__gamma': GAMMA_PARAM,
                                'classifier__kernel': KERNEL
                                }
            
            grid = GridSearchCV(pipe,param_grid,cv=kfold)
                        
            fit_time = time.time()            
            grid.fit(X_train,y_train)
            f_time = round(time.time()-fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = grid.score(X_train,y_train)
            print('Training Score {}'.format(score_train))
            score_test = grid.score(X_test,y_test)
            print('Testing Score {}'.format(score_test))
            training_accuracy_gridSearch.append(score_train)
            testing_accuracy_gridSearch.append(score_test)
            training_time_gridSearch.append(f_time)
            prediction_time_gridSearch.append(p_time)
                        
            best_params_classifiers['SVC'] = grid.best_params_
            print('Best Parameters {}'.format(grid.best_params_))
            
        elif classifier=='GaussianNB':
            pipe = Pipeline([
                    ('scaler',StandardScaler()),
                   
                    ('classifier',GaussianNB())
                    ])
            
            param_grid = {
                    'scaler':SCALER,
                    
                    }

            grid = GridSearchCV(pipe,param_grid,cv=kfold)
                        
            fit_time = time.time()            
            grid.fit(X_train,y_train)
            f_time = round(time.time()-fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = grid.score(X_train,y_train)
            print('Training Score {}'.format(score_train))
            score_test = grid.score(X_test,y_test)
            print('Testing Score {}'.format(score_test))
            training_accuracy_gridSearch.append(score_train)
            testing_accuracy_gridSearch.append(score_test)
            training_time_gridSearch.append(f_time)
            prediction_time_gridSearch.append(p_time)
                        
            best_params_classifiers['SVC'] = grid.best_params_
            print('Best Parameters {}'.format(grid.best_params_))
            
    return training_accuracy_gridSearch,testing_accuracy_gridSearch,best_params_classifiers,classifiers
