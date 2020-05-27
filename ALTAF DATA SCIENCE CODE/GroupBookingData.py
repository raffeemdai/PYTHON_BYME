"""
Load dataset from train/submission path.
Generate key.
Compute raw features.
"""
import numpy as np
import pandas as pd

class Loader:
    def __init__(self, load_type, path):
        self.load_type = load_type
        self.path = path
    
    def load(self):
        self.load_data_with_key()
        self.compute_raw_features()          
    
    """generate unique key (PNR#CreationDate) for each dataset."""
    def generate_key(self, ds, pnr_col, date_col):
        temp = ds[date_col].str.split("/", expand=True)
        keySet = ds[pnr_col] + "#" + temp[0].str.zfill(2)+temp[1].str.zfill(2)+"20"+ temp[2].str[-2:]
        keySet = keySet.str.strip().str.upper()
        return pd.DataFrame(keySet)
        
    """ Load, merge datasets and create only required features. """
    def load_data_with_key(self):
        # load raw data.        
        print(self.load_type + ' data  : loading..')
        self.ds_pnr_master = pd.read_csv(self.path + 'Group_Pnr.csv')
        self.ds_agent_master = pd.read_csv(self.path + 'Agent_Master.csv')
        self.ds_pnr_ticket = pd.read_csv(self.path + 'Group_BKNG_PNR_TICKET.csv')   
        self.ds_pnr_ssr = pd.read_csv(self.path  + 'GROUP_BKNG_PNR_SSR.csv')
        #self.ds_pnr_emd = pd.read_csv(path + 'GROUP_BKNG_PNR_EMD.csv')
        #self.ds_pnr_pax = pd.read_csv(path + 'GROUP_BKNG_PNR_PAX_HIST.csv')        
        #self.ds_pnr_tlt = pd.read_csv(path + 'GROUP_BKNG_PNR_TLT.csv')
        #ds_gb_details = pd.read_csv(path + 'Gwp_Details.csv')
                
        #ds_pnr_master.count()
        #ds_pnr_master['REC_LOCATOR'].value_counts().head(10) # repeated PNR.
        #ds_pnr_master.dtypes    
        #ds_pnr_master['PNR_KEY'].value_counts().head(10) # all unique now.
        self.ds_pnr_master['PNR_KEY'] =  self.generate_key(self.ds_pnr_master, "REC_LOCATOR", "PNR_CREATION_DATE")
        self.ds_pnr_ticket['PNR_KEY'] = self.generate_key(self.ds_pnr_ticket, "PNR", "PNR_CREATION_DATE")
        self.ds_pnr_ssr['PNR_KEY'] = self.generate_key(self.ds_pnr_ssr, "PNR", "BOOKING_DATE") 
        #self.ds_pnr_emd['PNR_KEY'] = generate_key(ds_pnr_emd, "PNR", "PNR_CREATION_DATE") 
        #self.ds_pnr_pax['PNR_KEY'] = generate_key(ds_pnr_pax, "PNR", "BOOKING_DATE")         
        #ds_pnr_tlt['PNR_KEY'] = generate_key(ds_pnr_tlt, "PNR", "BOOKING_DATE") 
        #ds_gb_details['PNR_KEY'] = generateKey(ds_gb_details, "PNR", "PNR_CREATION_DATE") 
        print(self.load_type + ' data  : loaded, key generated.')
        
    def compute_raw_features(self):
        self.ds_raw_features = self.ds_pnr_master.copy()
        
        # make raw features ready.
        all_dep_date = self.ds_raw_features['FIRST_DEPARTURE_DATE'].str.split('/', expand=True)        
        self.ds_raw_features['DAY'] = all_dep_date[0].astype(int)  
        self.ds_raw_features['MONTH'] = all_dep_date[1].astype(int)   
        self.ds_raw_features['QUARTER'] = (self.ds_raw_features['MONTH']-1)//3 + 1     
        self.ds_raw_features['OD'] = (self.ds_raw_features['FIRST_SEGMENTS_DTL'].str.split('-', expand=True))[0]
        self.ds_raw_features['PAX_BIN'] = pd.cut(self.ds_raw_features['PAX'], [0, 20, 30, 40, 50, 60, 70, 80, 90,100,200], 
                       labels=['0-20', '20-30', '30-40','40-50', '50-60','60-70','70-80','80-90','90-100', '100-200'])
        #print(self.ds_raw_features.columns)
        #print(self.ds_agent_master.columns)        
#        self.ds_raw_features = self.ds_raw_features.merge(self.ds_agent_master[['INC_AGENT_CODE', 'AGENT_TYPE', 'AGENT_CATEGORY']],
#                                               left_on='inc_agent_no', right_on='INC_AGENT_CODE', how = "left")
#        self.ds_raw_features['AGENT_TYPE'].fillna('UNKNOWN', inplace=True)
        
        if self.load_type == 'train':
            self.ds_raw_features['FLOWN_STATUS'] = np.where(self.ds_raw_features.PNR_TYPE == 'FLOWN', 1, 0)
        print(self.load_type + ' data  : raw features computed.')     
    
    