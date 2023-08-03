import os, sys

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_ordinal_columns = ['Credit_Mix']
            categorical_nominal_columns = ['Occupation', 'Payment_Behaviour', 'Payment_of_Min_Amount']
            numerical_columns = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts','Num_Credit_Card', 'Interest_Rate', 
                                 'Num_of_Loan','Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit','Num_Credit_Inquiries',
                                 'Outstanding_Debt', 'Credit_Utilization_Ratio','Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
            
            cat_ordinal_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='most_frequent')),
                ('oe', OrdinalEncoder(categories=[['Bad','Standard','Good']])),
                ('ss', StandardScaler(with_mean=False))
            ])

            cat_nominal_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder()),
                ('ss', StandardScaler(with_mean=False))
            ])

            num_pipeline = Pipeline([
                ('si', SimpleImputer(strategy='median')),
                ('ss', StandardScaler(with_mean=False))
            ])
            
            logging.info(f"Categorical ordinal column: {categorical_ordinal_columns}")
            logging.info(f"Categorical nominal column : {categorical_nominal_columns}")
            logging.info(f"Numerical columns : {numerical_columns}")

            preprocessor = ColumnTransformer([
                ('cat_ordinal_pipeline', cat_ordinal_pipeline, categorical_ordinal_columns),
                ('cat_nominal_pipeline', cat_nominal_pipeline, categorical_nominal_columns),
                ('num_pipeline', num_pipeline, numerical_columns)
            ])

            return preprocessor



        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read the dataset as dataframe")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = ['Credit_Score']

            input_feature_train_df = train_df.drop(columns=target_column_name)
            target_feature_train_df = train_df['Credit_Score']

            input_feature_test_df = test_df.drop(columns=target_column_name)
            target_feature_test_df = test_df['Credit_Score']

            logging.info("Applying preprocessor object to train and test set")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            le = LabelEncoder()
            target_feature_train_arr = le.fit_transform(target_feature_train_df)
            target_feature_test_arr = le.transform(target_feature_test_df)

            input_feature_train_transformed_df = pd.DataFrame(input_feature_train_arr)
            input_feature_train_transformed_df['target'] = target_feature_train_arr
            train_arr = np.array(input_feature_train_transformed_df)

            input_feature_test_transformed_df = pd.DataFrame(input_feature_test_arr)
            input_feature_test_transformed_df['target'] = target_feature_test_arr
            test_arr = np.array(input_feature_test_transformed_df)

            logging.info("Saved preprocessor object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            raise CustomException(e, sys)


