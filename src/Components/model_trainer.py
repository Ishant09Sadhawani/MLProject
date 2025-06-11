import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.components.logger import logging


from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifcats","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse']
                },
                "Linear Regression": {},
                "KNN": {
                    'n_neighbors': [3, 5, 7]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            best_model_score = max(model_report.values())
            best_model_index = list(model_report.values()).index(best_model_score)
            best_model_name = list(model_report.keys())[best_model_index]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("no best model found")
            logging.info(f"Best Found Model ob both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)

            score=r2_score(y_test,predicted)

            return score


        except Exception as e:
            raise CustomException(e,sys)