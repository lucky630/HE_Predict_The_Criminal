# -*- coding: utf-8 -*-
"""
@author: marios

"""
import numpy as np
import gc
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
#import lightgbm as lgb
    
def load_datas(filename):

    return joblib.load(filename)

def printfile(X, filename):

    joblib.dump((X), filename)
    
def printfilcsve(X, filename):

    np.savetxt(filename,X)

# compute all pairs of variables
def Make_2way(X, Xt):
    columns_length=X.shape[1]
    for j in range (0,columns_length):
        for d in range (j+1,columns_length):  
            print(("Adding columns' interraction %d and %d" % (j, d) ))
            new_column_train=X[:,j]+X[:,d]
            new_column_test=Xt[:,j]+Xt[:,d]    
            X=np.column_stack((X,new_column_train))
            Xt=np.column_stack((Xt,new_column_test))
    return X, Xt


def bagged_set(X,y,model, seed, estimators, xt, update_seed=True):
    
   baggedpred=[ 0.0  for d in range(0, (xt.shape[0]))]
   #loop for as many times as we want bags
   for n in range (0, estimators):
        X_t,y_c=shuffle(X,y, random_state=seed+n)
          
        if update_seed:
            model.set_params(random_state=seed + n)
        model.fit(X_t,y_c)
        preds=model.predict_proba(xt)[:,1] # predict probabilities
        # update bag's array
        for j in range (0, (xt.shape[0])):           
                baggedpred[j]+=preds[j]
   # divide with number of bags to create an average estimate            
   for j in range (0, len(baggedpred)): 
                baggedpred[j]/=float(estimators)
   # return probabilities            
   return np.array(baggedpred)      

 
def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("PERID,Criminal\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))      
     
def main():

        load_data=True          
        SEED=15
        Use_scale=False # if we want to use standard scaler on the data
        meta_folder="" 
        meta=["main_logit_1way","main_logit_2way2","main_logit_2way4","main_xgboos_count_2D","main_rgf_count_2D","main_rgf_2D","main_extratree_count_2D"]
        
        bags=1 
        y = np.loadtxt("train.csv", delimiter=',',usecols=[71], skiprows=1)
        if load_data:
            kk=0
            Xmetatrain=None
            Xmetatest=None     
            for modelname in meta :
                    mini_xtrain=np.loadtxt(meta_folder + modelname + '.train.csv')
                    mini_xtest=np.loadtxt(meta_folder + modelname + '.test.csv')   
                    mean_train=np.mean(mini_xtrain) 
                    mean_test=np.mean(mini_xtest)    # we calclaute the mean of the test set  predictions      
                    if kk==0:
                        kk=kk+1
                        Xmetatrain=mini_xtrain
                        Xmetatest=mini_xtest
                    else :
                        Xmetatrain=np.column_stack((Xmetatrain,mini_xtrain))
                        Xmetatest=np.column_stack((Xmetatest,mini_xtest))
            # we combine with the stacked features
            X=Xmetatrain
            X_test=Xmetatest
            #X,X_test= Make_3way(X, X_test)
            X,X_test= Make_2way(X, X_test)# add interractions
            # we print the pickles
            printfile(X,"xmetahome.pkl")  
            printfile(X_test,"xtmetahome.pkl")     

            X=load_datas("xmetahome.pkl")              
            print(("rows %d columns %d " % (X.shape[0],X.shape[1] )))                   
        else :

            X=load_datas("xmetahome.pkl")              
            print(("rows %d columns %d " % (X.shape[0],X.shape[1] )))
        
        outset="stacking" # Name of the model (quite catchy admitedly)
        number_of_folds =3 # repeat the CV procedure 5 times and save the holdout predictions


        print(("len of target=%d" % (len(y))))
        
        #model we are going to use
                       
        #model=lgb.LGBMClassifier()
        model=ExtraTreesClassifier(n_estimators=10000, criterion='entropy', max_depth=9,  min_samples_leaf=1,  n_jobs=30, random_state=1)        
        #model=LogisticRegression(C=0.01)
        train_stacker=[ 0.0  for k in range (0,(X.shape[0])) ] # the object to hold teh held-out preds
          
        mean_auc = 0.0 
        # cross validation model we are going to use
        kfolder=StratifiedKFold(y, n_folds=number_of_folds,shuffle=True, random_state=SEED)       
        i=0 # iterator counter
        print(("starting cross validation with %d kfolds " % (number_of_folds))) # some words to keep you engaged
        if number_of_folds>0:
            for train_index, test_index in kfolder:
                # get train (set and target variable)
                X_train = X[train_index]    
                y_train= np.array(y)[train_index]
                #talk about it
                print((" train size: %d. test size: %d, cols: %d " % (len(train_index) ,len(test_index) ,(X_train.shape[1]) )))

                if Use_scale:
                    stda=StandardScaler()            
                    X_train=stda.fit_transform(X_train)

                # get validation (set and target variable)
                X_cv= X[test_index]
                y_cv = np.array(y)[test_index]
                
                if Use_scale:
                    X_cv=stda.transform(X_cv)
                
                preds=bagged_set(X_train,y_train,model, SEED + i, bags, X_cv, update_seed=True)
               
                auc = roc_auc_score(y_cv,preds)                        
                
                print(("size train: %d size cv: %d AUC (fold %d/%d): %f" % (len(train_index), len(test_index), i + 1, number_of_folds, auc)))
             
                mean_auc += auc
                
               
                no=0
                for real_index in test_index:
                         train_stacker[real_index]=(preds[no])
                         no+=1
                i+=1 # update iterator
                
            if (number_of_folds)>0: # if we did cross validation the print the results
                mean_auc/=number_of_folds
                print((" Average AUC: %f" % (mean_auc) )) # keep calling it AUC (because you can)
                
            print((" printing train datasets in %s" % (meta_folder+ outset + "train.csv"))) # print the hold out predictions


        print ("start final modeling")
        
        if Use_scale:
            stda=StandardScaler()            
            X=stda.fit_transform(X)

        #load test data
        X_test=load_datas("xtmetahome.pkl")
        if Use_scale:
            X_test=stda.transform(X_test) 
            
        print(("rows %d columns %d " % (X_test.shape[0],X_test.shape[1] )))
        preds=bagged_set(X, y,model, SEED, bags, X_test, update_seed=True)          

        
        X_test=None
        gc.collect()
        
        print((" printing test datasets in %s" % (meta_folder + outset + "test.csv")))          
        
        save_results(preds, outset+"_submission_" +str(mean_auc) + ".csv")         

        print("Done.")  

if __name__=="__main__":
  main()
