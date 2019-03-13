class Myrandomforest(object):
    """best tool of random forest"""
    def __init__(self, config):
        if "params" in config:
            self.params = config["params"]
        else:
            self.set_default_params()
        if "X" in config and "y" in config:
            self.X = config["X"]
            self.y = config["y"]
        else:
            print ("lack train data")
            
    def train(self, is_classification = True):
        if is_classification:
            self.model = RandomForestClassifier(self.params)
        else:
            self.model = RandomForestRegressor(self.params)
        self.model.fit(self.X, self.y)
            
    def set_default_params(self):
        """
        n_estimators: int 默认10
        criterion: gini or entropy (default = gini)
        max_depth: (default = None)
        min_samples_split: 默认2
        min_samples_leaf: 默认1
        min_weight_fration_leaf: (default = 0)
        max_features: (default = auto)
        
        max_leaf_nodes: 最大叶节点个数
        min_impurity_split
        bootstrap: 是否又放回采样,默认True
        n_jobs: 默认1
        random_state
        verbose
        class_weight: None
        """
        params = {
            'n_estimators': 120,
            'max_depth': 5,
            'min_samples_split': 1,
            'min_samples': 1,
            'max_features': 'log2'
        }
    
    def find_best_params(self):
        """贪心坐标下降法"""
        pass
    
    def use_grid_search(self, is_classification = True):
        param_test = {
            'n_estimators': [120, 300, 500, 800, 1200],
            'max_depth': [5, 8, 15, 25, 30],
            'min_samples_split': [2,5,10,100],
            'min_samples_leaf': [1,2,5,10],
            'max_features': ['log2', 'sqrt']
        }
        if is_classification:
            gs = GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_test, scoring = 'roc_auc', cv = 5)
        else:
            gs = GridSearchCV(estimator = RandomForestRegressor(), param_grid = param_test, scoring = 'neg_mean_absolute_error', cv = 5)
        gs.fit(self.X, self.y)
        self.params = gs.best_params_
