"""
Load raw dataset.
Generate unique key (PNR#CREATION_DATE).
Compute flown rates for given attributes.
Resample data.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import GroupBookingData
from importlib import reload
reload(GroupBookingData)

#class Preprocess:
class Preprocess:    
    def __init__(self, train_path, submission_path, program_mode = 'dev'):
        self.program_mode = program_mode        
        self.train_path = train_path
        self.submission_path = submission_path   
        self.derived_features = []
        self.submission_data = None
    
    """ populate required features """    
    def populate_features(self, flown_rate_attributes) :    
        
        training_data = GroupBookingData.Loader(load_type = 'train', path= self.train_path)
        training_data.load()
        print('training data size : ' + str(training_data.ds_raw_features.shape))        
        submission_data = GroupBookingData.Loader(load_type = 'submission', path= self.submission_path)
        submission_data.load()        
        print('submission data size : ' + str(submission_data.ds_raw_features.shape))
        self.submission_data = submission_data
          
        flat_list_attributes = []
        for attrib in flown_rate_attributes:
            if isinstance(attrib, str):
                flat_list_attributes.append(attrib)
            else:
                flat_list_attributes += attrib
        flat_list_attributes = list(set(flat_list_attributes))
        
        train_features = ['PNR_KEY', 'REC_LOCATOR', 'PNR_CREATION_DATE'] + flat_list_attributes + ['PNR_TYPE','FLOWN_STATUS']
        ds_train_features = pd.DataFrame()
        ds_train_features = training_data.ds_raw_features[train_features]            
        # for submission
        submission_features = ['PNR_KEY', 'REC_LOCATOR', 'PNR_CREATION_DATE'] + flat_list_attributes
        ds_submission_features = pd.DataFrame()
        ds_submission_features = submission_data.ds_raw_features[submission_features] 
        
        self.derived_features = []
        for attrib in flown_rate_attributes:             
            attribute_flown_rate = self.get_flown_rate(data = training_data.ds_raw_features, attributes= attrib)
            attrib_list = [attrib] if not isinstance(attrib, (list,)) else attrib
            ds_train_features = ds_train_features.merge(attribute_flown_rate[attrib_list+['percent']], on = attrib_list, how = "left")
            self.derived_features.append(('_'.join(attrib_list) + '_FLOWN_RATE'))
            ds_train_features.rename(columns = {'percent' : '_'.join(attrib_list) + '_FLOWN_RATE' }, inplace = True)
            # for submission
            ds_submission_features = ds_submission_features.merge(attribute_flown_rate[attrib_list+['percent']], on = attrib_list, how = "left")
            ds_submission_features.rename(columns = {'percent' : '_'.join(attrib_list) + '_FLOWN_RATE' }, inplace = True)
        print('derived features')
        print(self.derived_features)      
        print('train, submission features populated.')
        return ds_train_features, ds_submission_features
    
    """ get flown rate based on given attributes """
    def get_flown_rate(self, data, attributes):
        if not isinstance(attributes, (list,)): 
            group_attributes = [attributes] + ['FLOWN_STATUS']
        else:
            group_attributes = attributes + ['FLOWN_STATUS']       
        df_summary = data[group_attributes]
        df_summary['flown_count'] = (df_summary.groupby(attributes)['FLOWN_STATUS'].transform('sum'))
        df_summary['total_count'] = (df_summary.groupby(attributes)['FLOWN_STATUS'].transform('count'))
        df_summary.drop('FLOWN_STATUS', axis = 1, inplace=True)
        df_summary.drop_duplicates(inplace=True)
        df_summary['percent'] =  df_summary['flown_count'] / df_summary['total_count']
        return df_summary   
    
    def resample(self, X, y):
        print("Before sampling : ")
        print(pd.Series(y).value_counts())
        smote = SMOTE(kind = "regular")
        X_sm, y_sm = smote.fit_sample(X, y) 
        print("After sampling : ")
        print(pd.Series(y_sm).value_counts())
        return X_sm, y_sm  
    
    def final_update(self, data):           
        ds_merged = data.copy()
        
        unique_pnr_ticket =  pd.DataFrame(self.submission_data.ds_pnr_ticket['PNR_KEY'].unique(), columns = ['PNR_KEY'])
        ds_merged = ds_merged.merge(unique_pnr_ticket, on = 'PNR_KEY', how='left', indicator='TICK')
        ds_merged['FLOWN_STATUS'] = np.where(ds_merged.TICK == 'both', 1, ds_merged['FLOWN_STATUS'])
            
        unique_pnr_ticket =  pd.DataFrame(self.submission_data.ds_pnr_ssr['PNR_KEY'].unique(), columns = ['PNR_KEY'])
        ds_merged = ds_merged.merge(unique_pnr_ticket, on = 'PNR_KEY', how='left', indicator='SR')
        ds_merged['FLOWN_STATUS'] = np.where(ds_merged.SR == 'both', 1, ds_merged['FLOWN_STATUS'])
        
        ds_merged.drop(['TICK', 'SR'], axis = 1, inplace=True)
        return ds_merged

       
          
        
        
        
        
        
        
        
    
    

