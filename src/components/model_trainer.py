import sys
import os
from dataclasses import dataclass

# directory reach
directory = os.path.dirname(os.path.abspath("__file__"))
print(directory)
# setting path
sys.path.append(os.path.dirname(os.path.dirname(directory)))


from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
	trained_model_file_path = os.path.join('artifacts',"model.pkl")


class ModelTrainer:
	def __init__(self):
		self.model_trainer_config = ModelTrainerConfig()

	def initiate_model_training(self,train_array,test_array):
		try:
			logging.info("Split training and test input data")
			
			X_train,y_train,X_test,y_test = (
				train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

			models = {
			"Random Forest":RandomForestRegressor(),
			"Decision Tree":DecisionTreeRegressor(),
			"Gradient Boosting":GradientBoostingRegressor(),
			"Linear Regression":LinearRegression(),
			"XGB Regressor":XGBRegressor(),
			"CatBoosting Regression":CatBoostRegressor(),
			"Ada Boosting Regression":AdaBoostRegressor()
			}

			params={
				"Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regression":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Ada Boosting Regression":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

			model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

			# best model score

			best_model_score = max(sorted(model_report.values()))

			best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
			
			best_model = models[best_model_name]

			if best_model_score < 0.6:
				raise CustomException("No best model.change params")

			logging.info("Basic Models trained and best model selected")
		
			save_object(file_path = self.model_trainer_config.trained_model_file_path,
				obj = best_model)

			predicted_best_model = best_model.predict(X_test)
			best_r2_score = r2_score(y_test,predicted_best_model)


		except Exception as e:
			raise CustomException(e,sys)


		return best_model,best_r2_score