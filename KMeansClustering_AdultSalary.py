#new=np.c_[]

#!/usr/bin/env python3
#usr/bin/bash -tt
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:19:55 2019

@author: anilchivukula
"""

#import the libraries
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from time import time


#import the dataset
dataset=pd.read_csv('adult.csv')

# Encode the categorical features as numbers
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

encoded_data, encoders = number_encode_features(dataset)


#X=dataset.iloc[:,:-1].values
#y=dataset.iloc[:,-1].values

X=encoded_data.iloc[:,:-1].values
y=encoded_data.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)


# Normalize feature data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

array_mean=[]
array_std=[]
array_var=[]

for axis in range(0,X_train.shape[1]):
    array_mean.append(X_train[:axis].mean())
    array_std.append(X_train[:axis].std())
    array_var.append(X_train[:axis].var())

print('\n\n\n','Array Of means: ',array_mean)
print('\n\n\n','Array Of std: ',array_std)
print('\n\n\n','Array Of var: ',array_var)

means_init = np.array([X[y == i].mean(axis=0) for i in range(2)])





#using the elbow method to find the optimal number of clusters//within cluster sum of squares is also called inertia in scikit learn
wcss=[]
for i in range(1,11):
    #check what is random initialization trap. we can write about it
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number Of Clusters')
plt.ylabel('WCSS')
plt.show()

X_plot=dataset.iloc[:,[10,12]].values

#X_plot=encoded_data.iloc[:,[4,10]].values

#applying k-means to the dataset
kmeans=KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X_plot)

#visualizing the clusters
plt.scatter(X_plot[y_kmeans==0,0],X_plot[y_kmeans==0,1],s=10,c='red',label='Cluster 1')
plt.scatter(X_plot[y_kmeans==1,0],X_plot[y_kmeans==1,1],s=10,c='blue',label='Cluster 2')
#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=5,c='yellow',label='Centroids')
plt.title('Clusters of Adult Salary')
plt.xlabel('capital gain')
plt.ylabel('hours per week')
plt.legend()
plt.show()

#accuracy = 1-accuracy_score(y, y_kmeans)



#function definition
def kmeans(X_train, X_test, y_train, y_test, init_means, no_iter = 300, component_list =[2,5,10,25,50,60], num_class = 2):
    

    array_homo =[]
    array_comp =[]
    array_sil =[]
    array_var =[]
    array_v_measure_score=[]
    array_adjusted_rand_score=[]
    array_adjusted_mutual_info_score=[]
    array_accuracy_score=[]
    array_time=[]
    
    for num_classes in component_list:
        time1=time()
        
        clf = KMeans(n_clusters= 2, init='k-means++')
        
        clf.fit(X_train)
        
        y_test_pred = clf.predict(X_test)
        
        
        y_test_pre=array(y_test_pred)
        
        y_test_pred=1-y_test_pre
        
          
        #Homogenity score on the test data
        homo = metrics.homogeneity_score(y_test, y_test_pred)
        array_homo.append(homo)
        
        
        #Completeness score
        comp = metrics.completeness_score(y_test, y_test_pred)
        array_comp.append(comp)
        
        #Silhoutette score
        sil = metrics.silhouette_score(X_test, y_test_pred, metric='euclidean')
        array_sil.append(sil)
        
        #Variance explained by the cluster
        #var=clf.score(X_test)
        #array_var.append(var)
        
        v_measure_score=metrics.v_measure_score(y_test,y_test_pred)
        array_v_measure_score.append(v_measure_score)
        
        adjusted_rand_score=metrics.adjusted_rand_score(y_test,y_test_pred)
        array_adjusted_rand_score.append(adjusted_rand_score)
        
        adjusted_mutual_score=metrics.adjusted_mutual_info_score(y_test,y_test_pred)
        array_adjusted_mutual_info_score.append(adjusted_mutual_score)
        
        accuracy_scores=float(sum(y_test_pred == y_test))/float(len(y_test))
        array_accuracy_score.append(accuracy_scores)
        
        time_value=time()-time1
        array_time.append(time_value)
        
    print('\n\n\n Homo Score',array_homo)    
    print('\n\n\n Completeness Score',array_comp)
    print('\n\n\n silhouette Score',array_sil)
    print('\n\n\n Variance Measure Score',array_v_measure_score)
    print('\n\n\n Adjusted Rand Score',array_adjusted_rand_score)
    print('\n\n\n Adjusted Mutual Info Score',array_adjusted_mutual_info_score)
    print('\n\n\n Accuracy Score',array_accuracy_score)
    print('\n\n\n Time Taken',array_time)
    

    #Generating plots
    fig4,ax4 = plt.subplots()
    ax4.plot(component_list, array_homo)
    ax4.plot(component_list, array_comp)
    ax4.plot(component_list, array_sil)
    plt.legend(['homogenity','completeness','silhoutette'])
    plt.xlabel('Number of clusters')
    plt.title('Performance evaluation scores for KMeans')


    #fig5, ax5 = plt.subplots()
    #ax5.plot(component_list, array_var)
    #plt.title('Variance explained by each cluster for KMeans')
    #plt.xlabel('Number of cluster')

    

    plt.show()


    #Training and testing accuracy for K = num_class

    #Assigning the initial means as the mean feature vector for the class
    init_mean = init_means
    clf = KMeans(n_clusters= num_class, init = init_mean)

    clf.fit(X_train)

    #Training accuracy
    #y_train_pred = clf.predict(X_train)
    #train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    #print('Training accuracy for KMeans for K = {}:  {}'.format(num_class, train_accuracy))

    #Testing accuracy
    #y_test_pred = clf.predict(X_test)
    #test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    #print('Testing accuracy for KMeans for K = {}:  {}'.format(num_class, test_accuracy))


    return component_list, array_homo, array_comp, array_sil, array_var



#function call
kmeans(X_train, X_test, y_train, y_test, init_means = means_init, component_list = [2,5,10,25,50,60], num_class = 2)





#(2) apply dimension reduction algorithms
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
PCA_data = PCA(n_components = 14,whiten=True)
temp = PCA_data.fit(X_train)


variance=temp.explained_variance_ratio_
var=np.cumsum(np.round(variance,decimals=3)*100)
print('\n\n\n','Eigen Values',var)

#temp1= temp.components_
PCA_data_trans = PCA_data.transform(X_train)
PCA_data_trans_test = PCA_data.transform(X_test)

means_init = np.array([PCA_data_trans[y_train == i].mean(axis=0) for i in range(2)])


kmeans(PCA_data_trans, PCA_data_trans_test, y_train, y_test, init_means =means_init, component_list = [2,5,10,25,50,60], num_class = 2)

array_inversemean=[]
array_inversestd=[]
array_inversevar=[]

inverse_x_train=PCA_data.inverse_transform(PCA_data_trans)
import scipy
for axis in range(0,inverse_x_train.shape[1]):
    array_inversemean.append(inverse_x_train[:,axis].mean())
    array_inversestd.append(inverse_x_train[:,axis].std())
    array_inversevar.append(inverse_x_train[:,axis].var())

print('\n\n\n','Inverse mean',array_inversemean)
print('\n\n\n','Inverse std',array_inversestd)
print('\n\n\n','Inverse var',array_inversevar)



ICA_data = FastICA(n_components = 14)
ICA_data.fit(X_train)
ICA_data_trans = ICA_data.transform(X_train)
ICA_data_trans_test = ICA_data.transform(X_test)

means_init = np.array([ICA_data_trans[y_train == i].mean(axis=0) for i in range(2)])

kmeans(ICA_data_trans, ICA_data_trans_test, y_train, y_test, init_means = means_init, component_list = [2,5,10,25,50,60], num_class = 2)

import scipy
array_kurt=[]
for each in range(0,ICA_data_trans.shape[1]):
    array_kurt.append(round(scipy.stats.kurtosis(ICA_data_trans[:,each],axis=0,fisher=True,bias=True),2))
print('\n\n\n', 'Kurtosis Array: ',array_kurt)


#n_components=3,eps=0.00001=65%


from sklearn.random_projection import GaussianRandomProjection
transformer = GaussianRandomProjection(n_components=14,eps=0.00001)
RP_data_trans = transformer.fit_transform(X_train)
RP_data_trans_test = transformer.fit_transform(X_test)

means_init = np.array([RP_data_trans[y_train == i].mean(axis=0) for i in range(2)])

kmeans(RP_data_trans, RP_data_trans_test, y_train, y_test, init_means = means_init, component_list = [2,5,10,25,50,60], num_class = 2)



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
transformer = LinearDiscriminantAnalysis(solver="svd",n_components = 14)
LDA_data_trans = transformer.fit_transform(X_train,y_train)
LDA_data_trans_test = transformer.fit_transform(X_test,y_test)

means_init = np.array([LDA_data_trans[y_train == i].mean(axis=0) for i in range(2)])
kmeans(LDA_data_trans, LDA_data_trans_test, y_train, y_test, init_means = means_init, component_list = [2,5,10,25,50,60], num_class = 2)


start_time=time()
######
##Apply neural network to one of the datasets with dimensionality reduced projected data
from sklearn.neural_network import MLPClassifier
neuralnet = MLPClassifier(solver='adam', alpha=0.000001,learning_rate_init=0.001,max_iter=100,verbose=0,hidden_layer_sizes=(7, 3))
classifierdata=neuralnet.fit(X_train,y_train)
y_train_pred = neuralnet.predict(X_train)
print('\n\n\n Training Accuracy for Just Neural Net: ',float(sum(y_train_pred == y_train))/float(len(y_train)))
y_test_pred = neuralnet.predict(X_test)
print('\n\n\n Test Accuracy For Just Neural Net: ',float(sum(y_test_pred == y_test))/float(len(y_test)))
print('\n\n Loss for just neural net',neuralnet.loss_)
print('\n\n Time taken for just neural network: ',time()-start_time)



start_time=time()
#neuralnet = MLPClassifier(solver='adam', alpha=0.00001,learning_rate_init=0.001,max_iter=100,verbose=0,hidden_layer_sizes=(7, 3))
neuralnet.fit(PCA_data_trans, y_train)  
y_train_PCA_pred = neuralnet.predict(PCA_data_trans)
print('\n\n\n Training Accuracy for PCA neural net',float(sum(y_train_PCA_pred == y_train))/float(len(y_train)))
y_test_pred_PCA = neuralnet.predict(PCA_data_trans_test)
print('\n\n\n Test Accuracy for PCA neural net',float(sum(y_test_pred_PCA == y_test))/float(len(y_test)))
print('\n\n Loss for PCA neural net',neuralnet.loss_)
print('\n\n Time taken for PCA neural network: ',time()-start_time)

start_time=time()
neuralnet.fit(ICA_data_trans, y_train)  
y_train_ICA_pred = neuralnet.predict(ICA_data_trans)
print('\n\n\n Training Accuracy for ICA neural net',float(sum(y_train_ICA_pred == y_train))/float(len(y_train)))
y_test_pred_ICA = neuralnet.predict(ICA_data_trans_test)
print('\n\n\n Test Accuracy for ICA neural net',float(sum(y_test_pred_ICA == y_test))/float(len(y_test)))
print('\n\n Loss for ICA neural net',neuralnet.loss_)
print('\n\n Time taken for ICA neural network: ',time()-start_time)


start_time=time()
neuralnet.fit(RP_data_trans, y_train)  
y_train_RP_pred = neuralnet.predict(RP_data_trans)
print('\n\n\n Training Accuracy for RP neural net',float(sum(y_train_RP_pred == y_train))/float(len(y_train)))
y_test_pred_RP = neuralnet.predict(RP_data_trans_test)
print('\n\n\n Test Accuracy for RP neural net',float(sum(y_test_pred_RP == y_test))/float(len(y_test)))
print('\n\n Loss for RP neural net',neuralnet.loss_)
print('\n\n Time taken for RP neural network: ',time()-start_time)


start_time=time()
neuralnet.fit(LDA_data_trans, y_train)  
y_train_LDA_pred = neuralnet.predict(LDA_data_trans)
print('\n\n\n Training Accuracy for LDA neural net',float(sum(y_train_LDA_pred == y_train))/float(len(y_train)))
y_test_pred_LDA = neuralnet.predict(LDA_data_trans_test)
print('\n\n\n Test Accuracy for LDA neural net',float(sum(y_test_pred_LDA == y_test))/float(len(y_test)))
print('\n\n Loss for LDA neural net',neuralnet.loss_)
print('\n\n Time taken for LDA neural network: ',time()-start_time)




