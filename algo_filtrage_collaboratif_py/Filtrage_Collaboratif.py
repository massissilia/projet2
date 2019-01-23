
# coding: utf-8

# In[1]:


#conda install -c conda-forge scikit-surprise : surprise package
#https://surprise.readthedocs.io/en/stable/getting_started.html
import numpy as np
import pandas as pd
import surprise
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV


# In[2]:


#import data
dataset = pd.read_excel("Donnees_Algo_Filtrage_Colla.xlsx")
dataset #3columns (users, products, order)


# In[3]:


#define a Reader object for Surprise to be able to parse the file
reader=surprise.Reader(rating_scale=(0.5,4.))
data=surprise.Dataset.load_from_df(dataset,reader)


# In[4]:


# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.20)
#factorisation algo svd++
algo=surprise.SVDpp()
# Train the algorithm on the trainset
algo.fit(trainset)
#predict ratings for the testset
predictions = algo.test(testset)
# Then compute RMSE
accuracy.rmse(predictions)


# In[5]:


###############DO RECOMMANDATIONS###############
iids=dataset['iid'].unique()#list of items
#list of items liked by each user
newDF = pd.DataFrame()
liste=dataset['uid'].unique()#list of users
iidi=[]
for i in range(0,len(liste)):
   iidi=dataset.loc[dataset['uid']==liste[i], 'iid']
   newDF = newDF.append(iidi, ignore_index = True)
liste = pd.DataFrame(liste)#transform liste to dataframe
liste=liste.rename(columns={0: 'user'},inplace=False)     
frames = [newDF,liste]
newDF = pd.concat(frames,axis=1)#dataframe of items liked by each user


# In[6]:


##items not scored by all users
newDF1=newDF.drop(['user'],axis=1)
newDF_to_pred= pd.DataFrame()
for i in range(0,len(newDF1)):
    iids_to_pred=np.setdiff1d(iids,newDF1.iloc[i])
    iids_to_pred = pd.DataFrame(iids_to_pred)#transform iids_to_pred to dataframe
    newDF_to_pred = newDF_to_pred.append(iids_to_pred, ignore_index = True)
newDF_to_pred=newDF_to_pred.rename(columns={0: 'item'},inplace=False) #item non scored


# In[7]:


##assosiate for each row of newDF_to_pred the user
z=dataset.groupby('uid').count().reset_index(drop=True).drop("rating", axis=1)
number_film_to_score=len(iids)-z['iid']
frame = [liste,number_film_to_score]
donnees = pd.concat(frame,axis=1)#dataframe for each user, the number of items to will be score
donnees
DATA = donnees.reindex(donnees.index.repeat(donnees.iid.apply(np.sum)))##repeat the number of ligne by using the iid column
DATA = DATA.reset_index(drop=True).drop("iid", axis=1)


# In[8]:


frame1 = [DATA,newDF_to_pred]
item_to_predict = pd.concat(frame1,axis=1)####dataframe users and items to predict


# In[9]:


#predict scores pour all users
for i in range(0,len(item_to_predict)-1):
    testset=[[i, iid, 4.] for (i,iid) in zip(item_to_predict['user'],item_to_predict['item'])]
predictions=algo.test(testset)### use alg svdpp
predictions[0]##see the first prediction


# In[10]:


#chaque prediction est un objet, on propose de convertir l'objet en tableau
pred_ratings=np.array([pred.est for pred in predictions])##scores predicted for items
frame2 = [item_to_predict,pd.DataFrame([pred_ratings]).T]
Prediction = pd.concat(frame2,axis=1).rename(columns={0: 'predict_score'},inplace=False) 
Prediction ##result


# In[11]:


#find the index of the maximum predicted rating for each user
Max_Prediction=Prediction.ix[Prediction.groupby(['user'], sort=False)['predict_score'].idxmax()][['user', 'item', 'predict_score']]
#recommander pour chaque user, le meilleur item


# In[12]:


Prediction.to_csv('Score_Prediction.xls', sep = '\t') 
Max_Prediction.to_csv('Max_Prediction.xls', sep = '\t')  
import shutil
Output_Filtrage_Collaboratif = open("Output_Filtrage_Collaboratif.xls", "w")
list_fichier =['Score_Prediction.xls','Max_Prediction.xls']
for i in list_fichier:
          shutil.copyfileobj(open(i, 'r'), Output_Filtrage_Collaboratif)
Output_Filtrage_Collaboratif.close()

