import pandas as pd
from sklearn import model_selection

'''
- binary classification
- multi class classification
- multi label classification
- single column classification
- multi column classification
- holdout
'''

class CrossValidation:
        def __init__(
                        self,
                        df,
                        target_cols,
                        shuffle,#with time-series data shuffle = False
                        problem_type="binary_classification",
                        multilabel_delimiter=",",
                        num_folds = 5,
                        random_state = 42
                ):
                self.dataframe = df
                self.target_cols = target_cols
                self.num_targets = len(target_cols)
                self.problem_type = problem_type
                self.num_folds = num_folds
                self.shuffle = shuffle
                # "," in front of shuffle
                self.random_state = random_state
                self.multilabel_delimiter = multilabel_delimiter

                if self.shuffle is True:
                        self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

                self.dataframe['kfold'] = -1

        def split(self):
                if self.problem_type in ("binary_classification","multiclass_classification"):
                        if self.num_targets != 1:
                                raise Exception("Invalid Number of targets for the Problem Type")
                        target = self.target_cols[0]
                        unique_values = self.dataframe[target].nunique()
                        if unique_values == 1:
                                raise Exception("Only one unique value found !")
                        elif unique_values > 1:
                                kf = model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                                    shuffle=False,#self.shuffle,
                                                                    random_state=None)#self.random_state)
                                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe,
                                                                                  y=self.dataframe[target].values)):
                                        #print(len(train_idx), len(val_idx))
                                        self.dataframe.loc[val_idx, 'kfold'] = fold

                elif self.problem_type in ("single_col_regression", "multi_col_regression"):
                        if self.num_targets != 1 and self.problem_type == "single_col_regression":
                                raise Exception("Invalid Number of targets for the Problem Type")
                        if self.num_targets < 2 and self.problem_type == "multi _col_regression":
                                raise Exception("Invalid Number of targets for the Problem Type")
                        kf = model_selection.KFold(n_splits=self.num_folds)
                        for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                                        #print(len(train_idx), len(val_idx))
                                        self.dataframe.loc[val_idx, 'kfold'] = fold

                elif self.problem_type.startswith("holdout_"):
                        holdout_percentage = int(self.problem_type.split("_")[1])
                        num_holdout_samples = int(len(self.dataframe) *  holdout_percentage / 100)
                        self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, 'kfold'] = 0
                        self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, 'kfold'] = 1

                elif self.problem_type == "multilabel_classification":
                        if self.num_targets != 1:
                                raise Exception("Invalid Number of targets for the Problem Type")
                        targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split                   (self.multilabel_delimiter)))
                        kf = model_selection.StratifiedKFold(n_splits=self.num_folds,shuffle=False,#self.shuffle,
                                                             random_state=None)#self.random_state)
                        for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe,y=targets)):
                            #print(len(train_idx), len(val_idx))
                            self.dataframe.loc[val_idx, 'kfold'] = fold 
                            
                else:
                        raise Exception("Problem Type Not Understood")

                return self.dataframe

if __name__ == '__main__':
        df = pd.read_csv("../input/train_mlclf.csv")
        #df = pd.read_csv("../input/train_screg.csv")
        #cv = CrossValidation(df, target_cols=["SalePrice"], problem_type="single_col_regression", shuffle=True)
        #cv = CrossValidation(df, target_cols=["target"], problem_type="holdout_10", shuffle=True)
        cv = CrossValidation(df, target_cols=["attribute_ids"], problem_type="multilabel_classification", shuffle=True,
                             multilabel_delimiter=" ")
        df_split = cv.split()
        print(df_split.head())
        print(df_split.kfold.value_counts())

        # multi label
        # ID    TARGET
        #  1    23,55
        #  2    32,76,78
        #  3    87,19