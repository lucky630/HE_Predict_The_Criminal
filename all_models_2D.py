
import numpy as np
from sklearn import  preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
import random
from catboost import CatBoostClassifier

SEED = 42  # always use a seed for randomized procedures

def load_data(filename, use_labels=True):

    # load column 1 to 8 (ignore last one)
    data = np.loadtxt(open( filename), delimiter=',',
                      usecols=list(range(1, 70)), skiprows=1)
    if use_labels:
        labels = np.loadtxt(open( filename), delimiter=',',
                            usecols=[71], skiprows=1)
    else:
        labels = np.zeros(data.shape[0])
    return labels, data


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("PERID,Criminal\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))


def bagged_set(X_t,y_c,model, seed, estimators, xt, update_seed=True):
    
   # create array object to hold predictions 
   baggedpred=[ 0.0  for d in range(0, (xt.shape[0]))]
   #loop for as many times as we want bags
   for n in range (0, estimators):
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
   
def printfilcsve(X, filename):
    np.savetxt(filename,X) 
    

# compute all pairs of variables
def Make_2way(X, Xt):
    columns_length=X.shape[1]
    for j in range (0,columns_length,2):
        for d in range (j+1,columns_length,random.randint(1,5)):  
            print(("Adding columns' interraction %d and %d" % (j, d) ))
            new_column_train=X[:,j]+X[:,d]
            new_column_test=Xt[:,j]+Xt[:,d]    
            X=np.column_stack((X,new_column_train))
            Xt=np.column_stack((Xt,new_column_test))
    return X, Xt
    

def main():
    filename="main_Catboost_count_2D" # nam prefix

    model=CatBoostClassifier(iterations=80, depth=3, learning_rate=0.1, loss_function='Logloss')
    #model = RGFClassifier(max_leaf=500,algorithm="RGF",test_interval=100, loss="LS")
    #model=lgb.LGBMClassifier(num_leaves=150,objective='binary',max_depth=6,learning_rate=.01,max_bin=400,auc='binary_logloss')

    # === load data in memory === #
    print("loading data")
    y, X = load_data('train.csv')
    y_test, X_test = load_data('test.csv', use_labels=False)
    
    X,X_test= Make_2way(X, X_test)# add interractions
    # === one-hot encoding === #
    #encoder = preprocessing.OneHotEncoder()
    #encoder.fit(np.vstack((X, X_test)))
    #X = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)
    #X_test = encoder.transform(X_test)

    train_stacker=[ 0.0  for k in range (0,(X.shape[0])) ]

    # === training & metrics === #
    mean_auc = 0.0
    bagging=2 # number of models trained with different seeds
    n = 2  # number of folds in strattified cv
    kfolder=StratifiedKFold(y, n_folds= n,shuffle=True, random_state=SEED)     
    i=0
    for train_index, test_index in kfolder:
        # creaning and validation sets
        X_train, X_cv = X[train_index], X[test_index]
        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
       
        preds=bagged_set(X_train,y_train,model, SEED , bagging, X_cv, update_seed=True)   
        
        roc_auc = roc_auc_score(y_cv, preds)
        print("AUC (fold %d/%d): %f" % (i + 1, n, roc_auc))
        mean_auc += roc_auc
        
        no=0
        for real_index in test_index:
                 train_stacker[real_index]=(preds[no])
                 no+=1
        i+=1

    mean_auc/=n
    print((" Average AUC: %f" % (mean_auc) ))
    print (" printing train datasets ")
    printfilcsve(np.array(train_stacker), filename + ".train.csv")          

    # === Predictions === #
    # When making predictions, retrain the model on the whole training set
    preds=bagged_set(X, y,model, SEED, bagging, X_test, update_seed=True)  

    #create submission file 
    printfilcsve(np.array(preds), filename+ ".test.csv")  
    #save_results(preds, filename+"_submission_" +str(mean_auc) + ".csv")

if __name__ == '__main__':
    main()
