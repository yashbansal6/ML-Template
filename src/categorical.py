from sklearn import preprocessing

"""
- label encoding
- one hot encoding
- binarization
"""

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df = pandas df
        catagorical_features = list of col names => ['bin_1', 'nom_1', 'ord_1']
        encoding_type = label, binary, onehot
        handle_na = True / False
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None
        
        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:,c] = self.df.loc[:,c].astype(str).fillna("-999999")
                #This imaginary value should not be present in dataset
        
        #make copy after handling NaN
        self.output_df = self.df.copy(True)
        
    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:,c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df
    
    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            # transform returns array
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:,j]
            self.binary_encoders[c] = lbl
        return self.output_df    
    
    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        return ohe.transform(self.df[self.cat_feats].values)
            
    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        else:
            raise Exception("Encoding Type not understood")
    
    # for test dataset    
    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:,c] = dataframe.loc[:,c].astype(str).fillna("-999999")
        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:,c] = lbl.transform(dataframe[c].values)
            return dataframe
        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)
                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:,j]
            return dataframe
        
if __name__ == "__main__":
    import pandas as pd
    '''
    #label
    
    df = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")
    cols = [c for c in df.columns if c not in ["id", "target"]]
    print(cols)
    
    cat_feats = CategoricalFeatures(df, categorical_features=cols, encoding_type="label", handle_na=True)
    output_df = cat_feats.fit_transform()
    print(output_df.head())
    '''
    '''
    #binary - very long processing time + huge space
    
    df = pd.read_csv("../input/cat-in-the-dat-ii/train.csv").head(500)
    cols = [c for c in df.columns if c not in ["id", "target"]]
    print(cols)
    
    cat_feats = CategoricalFeatures(df, categorical_features=cols, encoding_type="binary", handle_na=True)
    output_df = cat_feats.fit_transform()
    print(output_df.head())
    '''
    '''
    # test dataset - works for binary but not for label due to previously unseen labels
    
    df = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")
    df_test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")
    
    cols = [c for c in df.columns if c not in ["id", "target"]]
    print(cols)
    
    cat_feats = CategoricalFeatures(df, categorical_features=cols, encoding_type="label", handle_na=True)
    train_transformed = cat_feats.fit_transform()
    test_transformed = cat_feats.transform(df_test)
    print(test_transformed)
    '''
    '''
    #to solve the issue of previous unseen labels, combine train and test dataset
    
    df = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")
    df_test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")
    
    train_idx = df["id"].values
    test_idx = df_test["id"].values
    
    df_test["target"] = -1 
    full_data = pd.concat([df, df_test])
    
    cols = [c for c in df.columns if c not in ["id", "target"]]
    print(cols)
    cat_feats = CategoricalFeatures(full_data, categorical_features=cols, encoding_type="label", handle_na=True)
    full_data_transformed = cat_feats.fit_transform()
    
    train_df = full_data_transformed[full_data_transformed["id"].isin(train_idx)].reset_index(drop=True)
    test_df = full_data_transformed[full_data_transformed["id"].isin(test_idx)].reset_index(drop=True)
    
    print(train_df.shape)
    print(test_df.shape)
    '''
    #one hot encoder - requires concatenation - requires to remove NaN
    
    df = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")
    df_test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")
    
    train_len = len(df)
    
    df_test["target"] = -1 
    full_data = pd.concat([df, df_test])
    
    cols = [c for c in df.columns if c not in ["id", "target"]]
    print(cols)
    cat_feats = CategoricalFeatures(full_data, categorical_features=cols, encoding_type="ohe", handle_na=True)
    full_data_transformed = cat_feats.fit_transform()
    
    train_df = full_data_transformed[:train_len, :]
    test_df = full_data_transformed[train_len:, :]
    
    print(train_df.shape)
    print(test_df.shape)