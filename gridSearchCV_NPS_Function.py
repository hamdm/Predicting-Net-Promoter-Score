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
import time


kfold = KFold(n_splits=5,shuffle=True,random_state=0)
SCALER = [None,StandardScaler()]
FEATURES = [40,32,28,26,24]
REDUCER__N_COMPONENTS = [2,6,10,14,18,22]

classifiers = ['KNeighbors','LogisticRegression','LinearSVC',
               'DecisionTree','RandomForest','GradientBoosting','SVC']



def gridSearch_FS(X_train,X_test,y_train,y_test):


    training_accuracy_fs = []
    testing_accuracy_fs = []
    training_time_fs = []
    prediction_time_fs = []
    best_params_classifiers_fs = {}
    
    for classifier in classifiers:
        if classifier=='KNeighbors':

            NEIGHBORS = range(1,11)

            pipe = Pipeline([
                    ('scaler',StandardScaler()),
                    ('selector',RFE(RandomForestClassifier(n_estimators=10,random_state=42))),
                    ('classifier',
                     KNeighborsClassifier())])
    
            param_grid = {
                                'scaler': SCALER,
                                'selector__n_features_to_select':FEATURES,
                                'classifier__n_neighbors': NEIGHBORS
                                }
            knn_grid = GridSearchCV(pipe,param_grid,cv=kfold)
            
            
            fit_time = time.time()            
            knn_grid.fit(X_train,y_train)
            f_time = round(time.time()-fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = knn_grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = knn_grid.score(X_train,y_train)
            score_test = knn_grid.score(X_test,y_test)
            training_accuracy_fs.append(score_train)
            testing_accuracy_fs.append(score_test)
            training_time_fs.append(f_time)
            prediction_time_fs.append(p_time)
                        
            best_params_classifiers_fs[classifier] = knn_grid.best_params_

        elif classifier=='LogisticRegression':

            PENALTY = ['l2','l1']
            C_PARAM_LOG = [0.001,0.01,0.1,1,10,100]
            
            pipe = Pipeline([
                    ('scaler',StandardScaler()),
                    ('selector',RFE(RandomForestClassifier(n_estimators=10,random_state=42))),
                   
                    ('classifier',LogisticRegression())
                    ])
            
            param_grid = {
                    'scaler':SCALER,
                    'selector__n_features_to_select':FEATURES,
                    
                    'classifier__penalty':PENALTY,
                    'classifier__C': C_PARAM_LOG,
                    }
            lr_grid = GridSearchCV(pipe,param_grid,cv=kfold)
            
            fit_time = time.time()            
            lr_grid.fit(X_train,y_train)
            f_time = round(time.time() - fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = lr_grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                
            score_train = lr_grid.score(X_train,y_train)
            score_test = lr_grid.score(X_test,y_test)
            training_accuracy_fs.append(score_train)
            testing_accuracy_fs.append(score_test)
            training_time_fs.append(f_time)
            prediction_time_fs.append(p_time)

            best_params_classifiers_fs[classifier] = lr_grid.best_params_

        elif classifier=='LinearSVC':
            C_PARAM_LSVC = [0.001,0.01,0.1,1,10,100]
            
            pipe = Pipeline([
                                ('scaler',StandardScaler()),
                                ('selector',RFE(RandomForestClassifier(n_estimators=10,random_state=42))),
                                
                                ('classifier',LinearSVC())
                                ])
                        
            param_grid = {
                                'scaler':SCALER,
                                'selector__n_features_to_select':FEATURES,
                               
                             
                                'classifier__C': C_PARAM_LSVC,
                                }
            
            lsvc_grid = GridSearchCV(pipe,param_grid,cv=kfold)
                        
            fit_time = time.time()            
            lsvc_grid.fit(X_train,y_train)
            f_time = round(time.time()-f_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = lsvc_grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = lsvc_grid.score(X_train,y_train)
            score_test = lsvc_grid.score(X_test,y_test)
            training_accuracy_fs.append(score_train)
            testing_accuracy_fs.append(score_test)
            training_time_fs.append(f_time)
            prediction_time_fs.append(p_time)
                        
            best_params_classifiers_fs[classifier] = lsvc_grid.best_params_
            
        elif classifier == 'DecisionTree':
            CRITERION = ['gini','entropy']
            SPLITTER = ['best','random']
            MIN_SAMPLES_SPLIT = [2,4,6,8]
            CLASS_WEIGHT = ['balanced',None]
                        
            pipe = Pipeline([
                                ('scaler',StandardScaler()),
                                ('selector',RFE(RandomForestClassifier(n_estimators=10,random_state=42))),
                               
                                ('classifier',DecisionTreeClassifier())
                                ])
                        
            param_grid = {
                            'scaler': SCALER,
                            'selector__n_features_to_select':FEATURES,
                            
                            'classifier__criterion': CRITERION,
                            'classifier__splitter': SPLITTER,
                            'classifier__min_samples_split': MIN_SAMPLES_SPLIT,
                            'classifier__class_weight': CLASS_WEIGHT,
                        }
            
            dt_grid = GridSearchCV(pipe,param_grid,cv=kfold)
                        
            fit_time = time.time()            
            dt_grid.fit(X_train,y_train)
            f_time = round(time.time()-fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = dt_grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = dt_grid.score(X_train,y_train)
            score_test = dt_grid.score(X_test,y_test)
            training_accuracy_fs.append(score_train)
            testing_accuracy_fs.append(score_test)
            training_time_fs.append(f_time)
            prediction_time_fs.append(p_time)
                        
            best_params_classifiers_fs[classifier] = dt_grid.best_params_
        
        elif classsifier=='RandomForest':
            ESTIMATOR = [30,70,90,110]
            CRITERION_RFC = ['gini','entropy']
            rfc_FEATURES = ['auto',8,10,12]
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
                            'classifier__max_features': rfc_FEATURES,
                            'classifier__min_samples_split': MIN_SAMPLES_SPLIT,
                            'classifier__class_weight': CLASS_WEIGHT,
                        }
            
            rt_grid = GridSearchCV(pipe,param_grid,cv=kfold)
                        
            fit_time = time.time()            
            rt_grid.fit(X_train,y_train)
            f_time = round(time.time()-fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = rt_grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = rt_grid.score(X_train,y_train)
            score_test = rt_grid.score(X_test,y_test)
            training_accuracy_fs.append(score_train)
            testing_accuracy_fs.append(score_test)
            training_time_fs.append(f_time)
            prediction_time_fs.append(p_time)
                        
            best_params_classifiers_fs[classifier] = rt_grid.best_params_
            
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
            
            gb_grid = GridSearchCV(pipe,param_grid,cv=kfold)
                        
            fit_time = time.time()            
            gb_grid.fit(X_train,y_train)
            f_time = round(time.time()-fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = gb_grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = gb_grid.score(X_train,y_train)
            score_test = gb_grid.score(X_test,y_test)
            training_accuracy_fs.append(score_train)
            testing_accuracy_fs.append(score_test)
            training_time_fs.append(f_time)
            prediction_time_fs.append(p_time)
                        
            best_params_classifiers_fs[classifier] = gb_grid.best_params_
        
        elif classifier=='SVC':
            C_PARAM = [0.001,0.01,0.1,1,10,100]
            GAMMA_PARAM = [0.001,0.01,0.1,1,10,100]
            FEATURES = [40,32,28,26,24,18,14]
            KERNEL = ['rbf','sigmoid']
                        
            pipe = Pipeline([
                                ('scaler',StandardScaler()),
                                ('selector',RFE(RandomForestClassifier(n_estimators=10,random_state=42))),
                               
                                ('classifier',SVC())
                                ])
                        
            param_grid = {
                                'scaler':SCALER,
                                'selector__n_features_to_select':FEATURES,
                              
                                'classifier__C': C_PARAM,
                                'classifier__gamma': GAMMA_PARAM,
                                'classifier__kernel': KERNEL
                                }
            
            svc_grid = GridSearchCV(pipe,param_grid,cv=kfold)
                        
            fit_time = time.time()            
            svc_grid.fit(X_train,y_train)
            f_time = round(time.time()-fit_time,3)
            print('training time for {} : {}'.format(classifier,f_time))
            pred_time = time.time()
            predictions = svc_grid.predict(X_test)
            p_time = round(time.time()-pred_time,3)
            print('predict time: {}'.format(p_time))
                        
            score_train = svc_grid.score(X_train,y_train)
            score_test = svc_grid.score(X_test,y_test)
            training_accuracy_fs.append(score_train)
            testing_accuracy_fs.append(score_test)
            training_time_fs.append(f_time)
            prediction_time_fs.append(p_time)
                        
            best_params_classifiers_fs[classifier] = svc_grid.best_params_
            
    return training_accuracy_fs,testing_accuracy_fs,best_params_classifiers_fs,classifiers
