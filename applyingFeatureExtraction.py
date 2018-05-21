from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def apply_MinMax(X_train,X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled,X_test_scaled


def explained_variance(X_train,X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_ss = scaler.transform(X_train)
    X_test_ss = scaler.transform(X_test)
    
    pca = PCA(n_components=None)
    pca.fit(X_train_ss)
    X_train_pca = pca.transform(X_train_ss)
    X_test_pca = pca.transform(X_test_ss)
    explained = pca.explained_variance_ratio_ 
    
    return X_train_pca,X_test_pca,explained


def components_to_PCA(explained_var,required_variance):
    variance = 0
    for i in range(0,63):
        if variance <= required_variance:
            variance = variance + explained_var[i]
            print(variance)
        else:
            print('You need first {} principal components to achieve {} variance'.format(i,required_variance))
            break

def apply_PCA(X_train,X_test,n_components):

    '''
    First applying StandardScaler to bring features to unit variance, and then
    doing the Principal Component Analysis
    '''
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)



    pca = PCA(n_components=n_components)
    pca.fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return X_train_pca,X_test_pca

'''
def apply_lda(X_train,X_test):
    
    LDA - Linear Discriminant Analysis, separates the most the classes
    of the dependent variable
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lda = LDA()
    X_train_lda = lda.fit_transform(X_train_scaled,y_train)
    X_test_lda = lda.transform(X_test_scaled)
    
    return X_train_da,X_test_lda
'''
