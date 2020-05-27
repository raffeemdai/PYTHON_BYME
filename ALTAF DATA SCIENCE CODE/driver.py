"""
1. Call GroupBookingPreprocessor to load and preprocess data.
2. Call GroupBookingClassifier to predict.
"""
import numpy as np
import GroupBookingPreprocessor
import GroupBookingClassifier
from importlib import reload
import argparse
import warnings
warnings.filterwarnings('ignore')

# single function to call load, preprocess, classify and save predictions.
def load_preprocess_classify(train_path, submission_path): 
    
    train_path = ".\\train\\"
    submission_path = ".\\test\\"
    
    print("\nTrain data path : " + train_path)
    print("Valiation data path : " + submission_path)
    flown_rate_attributes = ['inc_agent_no', 
                             'FIRST_SEGMENTS_DTL',                             
                             'MONTH', 
                             'PAX_BIN',                         
                             ['MONTH','FIRST_SEGMENTS_DTL'], 
                             ['inc_agent_no', 'MONTH'],
                             ['inc_agent_no', 'FIRST_SEGMENTS_DTL'],
                             ['inc_agent_no', 'FIRST_SEGMENTS_DTL', 'MONTH']]
    preprocessor = GroupBookingPreprocessor.Preprocess(train_path, submission_path)                                               
    ds_train_features, ds_submission_features = preprocessor.populate_features(flown_rate_attributes)
     
    ds_submission_features = ds_submission_features.fillna(ds_submission_features.mean())
    print('submission data - any null ?- ' + str(ds_submission_features.isnull().values.any()))
    #print('Columns with NA :- ')
    #print(ds_submission_features.isnull().sum())
    
    input_features = preprocessor.derived_features
    target_feature = 'FLOWN_STATUS'
    X = ds_train_features[input_features].values
    y = ds_train_features[target_feature].values
    print("training X size  : " + str(X.shape))
    print("training y size  : " + str(X.shape))
    # oversampling to balance dataset.
    #X_sampled, y_sampled  = X, y
    X_sampled, y_sampled = preprocessor.resample(X, y)
    
    # spilt train and test sets
    from sklearn.model_selection import train_test_split
    #from sklearn.metrics import accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size = 0.0001, random_state = 42)
    
    # classifiers.
    training_classifier = GroupBookingClassifier.Classify()    
    classifier, y_pred, train_cv, test_cv = training_classifier.runOne(X_train, y_train, X_test, y_test, modelName="XGB")
#    from sklearn.metrics import classification_report, confusion_matrix
#    accuracy_score(y_test, y_pred)
#    target_classes = ['CANCELLED', 'FLOWN']
#    print('classification report')
#    print(classification_report(y_test, y_pred, target_names=target_classes))
#    print('confustion matrix')
#    print(confusion_matrix(y_test, y_pred))
    
    """ submission/prediction ..................................... """
    X_submission = ds_submission_features[input_features].values
    y_submission_ped = classifier.predict(X_submission)
    ds_submission_final = ds_submission_features[['PNR_KEY','REC_LOCATOR', 'PNR_CREATION_DATE']]
    ds_submission_final.rename(columns = {'REC_LOCATOR' : 'PNR',
                                           'PNR_CREATION_DATE' : 'BOOKING DATE'
                                           }, inplace = True)
    ds_submission_final['FLOWN_STATUS'] = y_submission_ped
    ds_submission_final = preprocessor.final_update(ds_submission_final)
    
    ds_submission_final['PNR TYPE'] = np.where(ds_submission_final.FLOWN_STATUS == 1, 'FLOWN', 'CANCELLED')
    ds_submission_final.drop(['PNR_KEY', 'FLOWN_STATUS'], axis = 1, inplace=True)
    ds_submission_final.to_csv(submission_path + 'Output.csv', index=False)
    print('Output written to : ' + submission_path + 'Output.csv')  
#    disable before submission.
#    import pandas as pd
#    flown_actuals = pd.read_csv(submission_path + 'actuals.csv')
#    print('Accuracy on staging unseen data')
#    print(accuracy_score(ds_submission_final['PNR TYPE'], flown_actuals['PNR_TYPE']))
    print('End.')
    
if __name__ == "__main__":
    reload(GroupBookingPreprocessor)
    reload(GroupBookingClassifier)
    parser = argparse.ArgumentParser(description='Group Booking Materialization Hackathon.')
    parser.add_argument('-t','--trainPath', help='Path to training data.', required=True)
    parser.add_argument('-v','--validationPath', help='Path to validatio data', required=True)
    args = vars(parser.parse_args())   
    train_path=args['trainPath'] + '\\'
    submission_path =  args['validationPath'] + '\\'    
     
    load_preprocess_classify(train_path, submission_path)


