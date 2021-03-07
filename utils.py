import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LinearlyCorrelatedFeatures(BaseEstimator, TransformerMixin):
    """
    A custom transformer that transforms X to have only the features that are linearly correlated to y. The threshold 
    for the amount of correlation can be controlled.

    Pipeline will run this: self.fit(X, y, **fit_params).transform(X); Thus need X,y for fit() and it should return
    "self" thus making it possible it chaing like that, ie self.fit.transform. Need only X for transform()
    """

    def __init__(self, correlation_threshold=0.1):

        self.correlation_threshold = correlation_threshold
        self.column_filter = None

    def to_numpy(self, obj):
        """
        Convert an DataFrame to numpy if its already not one.
        """
        # from df to np
        if isinstance(obj, pd.core.frame.DataFrame) or isinstance(obj, pd.core.series.Series):
            return obj.values
        # no change
        elif isinstance(obj, np.ndarray):
            return obj
        # if not df or numpy, throw error:
        else:
            print(type(obj))
            raise ValueError("Numpy array or Dataframe should be passed.") 

    def get_correlated_features(self, dataset):
        """
        Returns a list of boolean (filter); True for the columns that have high linear corrleation with the last column
        of the numpy array (dataset = X+y; thus last column=y);
        args:
            dataset: m x n+1 array; 
        """
        y_column_id = dataset.shape[1]-1
        # gotta get the correlation between the columns; so transpose
        corr = np.corrcoef(dataset.T) 

        # get the DEATH_EVENT Row, then get a map of columns with abs values > threshold
        column_filter = abs(corr[y_column_id:]) > self.correlation_threshold

        # convert to list and choose index 0 since its nd array
        column_filter = column_filter.tolist()[0]

        # delete last element as it denotes "y"
        column_filter = column_filter[:-1]

        return column_filter

    def fit(self, X, y):
        """
        Selects the correlated features and saves it in "final_col"
        #todo: add checks for array/df size; if 'y' is vector; if rows of 'X' = rows of 'y'
        args:
            X: m x n DataFrame or ndarry
            y: m x 1 Series or ndarry
        """
        X = self.to_numpy(X)
        y = self.to_numpy(y)
        
        # merge them to a single array
        dataset = np.column_stack([X,y])
        self.column_filter = self.get_correlated_features(dataset)
        # should return self for chaining with .transform()
        return self
        
    def transform(self, X):
        """
        Returns a new new X with only the columns in "column_filter"
        """
        X = self.to_numpy(X)
        fil = np.array(self.column_filter)
        return X[:, fil]


class HeartFailurePrediction():
    """A Class that compares the effects of using without feature selection and using the Highly Correlated Features.
    """
    def __init__(self, dataset, scoring, cv, n_jobs, verbose, print_grid_scores, print_prediction_results, random_state):
        self.dataset = dataset
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.print_grid_scores = print_grid_scores
        self.print_prediction_results = print_prediction_results
        self.all_test_scores = {}
        self.X_train = None
        self.X_train_sel = None
        self.X_test = None
        self.X_test_sel = None
        self.y_train = None
        self.y_test = None
        self.random_state = random_state

    def preprocessing(self):
        # Seperating X,y
        output_col = "DEATH_EVENT"
        X = self.dataset.drop([output_col], axis=1)
        y = self.dataset[output_col]

        # Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,random_state=42, test_size=0.2)

        # Creating Preprocessing pipeling
        preprocessing = Pipeline([
            ("corr_features", LinearlyCorrelatedFeatures(correlation_threshold=0.1)),
            ("std_scalar", StandardScaler()),
        ])

        # Preprocessing is done only using train set; thus there is no information leakage from test_set
        preprocessing.fit(self.X_train,self.y_train)

        # Transforming them; basically we did feature selection.
        self.X_train_sel = preprocessing.transform(self.X_train)
        self.X_test_sel = preprocessing.transform(self.X_test)

    def main_wrapper(self, model,param, model_name):
        """
        Compares a model with and without selectin the highly correlated feature
        """
        self.preprocessing()
        temp = []
        p = self.print_grid_scores or self.print_prediction_results

        if p:
            print("-"*70)
            print("Without Feature Selection")
            print("-"*70)
        y_pred, gird_scores, test_scores = self.execute_single_model(model,
                                                                    param,
                                                                    self.X_train,
                                                                    self.y_train,
                                                                    self.X_test,
                                                                    self.y_test)
        if p:
            print("-"*70)
            print("With Feature Selection")
            print("-"*70)
        y_pred_sel, gird_scores_sel, test_scores_sel = self.execute_single_model(model,
                                                                    param,
                                                                    self.X_train_sel,
                                                                    self.y_train,
                                                                    self.X_test_sel,
                                                                    self.y_test)


        # Store the test scores:
        temp.append(test_scores)
        temp.append(test_scores_sel)
        self.all_test_scores[model_name] = temp
        
        # Confusion matrix:
        sns.set_style("white")
        f, axes = plt.subplots(1, 2, figsize=(15, 10), sharey='row')
        
        cf_mat = confusion_matrix(self.y_test, y_pred)
        cf_mat_sel = confusion_matrix(self.y_test, y_pred_sel)
        
        disp = ConfusionMatrixDisplay(cf_mat)
        disp_sel = ConfusionMatrixDisplay(cf_mat_sel)
        
        disp.plot(ax=axes[0])
        disp.ax_.set_title("Without Feature Selection")
        disp.im_.colorbar.remove()
        
        disp_sel.plot(ax=axes[1])
        disp_sel.ax_.set_title("With Feature Selection")
        disp_sel.im_.colorbar.remove()
        
    def execute_single_model(self, model, param, X_train, y_train, X_test, y_test):
        """
        Runs GridSearchCV to get a best estimator and predicts the X_test on it. 
        
        Returns:
        y_pred = Prediciton of the model on X_test
        grid_scores = various scores obtained from GridSearchCV
        test_scores = various scores obtained on the test set's prediciton
        """
        # fit the grid and get the gird obj
        grid = self.fit_grid(model, param, X_train, y_train)

        # use the grid obj to get the grid scores
        grid_scores = self.get_grid_scores(grid, self.print_grid_scores)

        # Since refit=True, the model is refitted using the best parameter and is available via best_estimator
        best_model = grid.best_estimator_

        # predict on the test_set
        y_pred = best_model.predict(X_test)

        # get scores on the preiciton on test_set
        test_scores = self.get_prediction_results(y_test, y_pred, self.print_prediction_results)

        return y_pred, grid_scores, test_scores

    def fit_grid(self, model, param, X, y):
        """
        Returns the GridSearch obj after fitting it on the given X and y
        """
        grid = GridSearchCV(model, param, refit=True, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs, verbose=self.verbose)
        grid.fit(X,y)

        return grid


    def get_grid_scores(self, grid, p):
        """
        Returns a dict with scores on each run; Can be used to print (p=True) them or used for anlysis later.
        Structure of dict:
        {
            best_score: {score:, param:}
            other_scores: [[score1, param1], [score2, param2], [], [] ]
        }
        """
        scores = {"best_score":{}, "other_scores": []}
        best_mean_score = 0
        best_param = None
        grid_scores = grid.cv_results_
        
        for mean_score, params in zip(grid_scores["mean_test_score"], grid_scores["params"] ):
            # get best score:
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_param = params
            
            # store other_scores
            temp = [mean_score, params]
            scores["other_scores"].append(temp)
            if p:
                print("%.4f" % mean_score, params)
        
        # store best_score
        scores["best_score"]["score"] = best_mean_score
        scores["best_score"]["param"] = best_param
        if p:
            print(f"Best Param:", "%.4f" % best_mean_score, best_param)
        
        
        return scores

    def get_prediction_results(self, y, y_pred, p):
        """
        Returns a dict with different scores like accuracy, recall, precision, f1.
        p : print results
        """
        scores = {}
        acc = accuracy_score(y, y_pred)*100
        pre = precision_score(y, y_pred)*100
        rec = recall_score(y, y_pred)*100
        f1 = f1_score(y, y_pred)*100
        
        scores['Accuracy'] = acc
        scores['Precision'] = pre
        scores['Recall'] = rec
        scores['F1 Score'] = f1
        if p:
            print('Accuracy Score : ', "{:.2f}%".format(acc))
            print('Precision Score : ', "{:.2f}%".format(pre))
            print('Recall Score : ', "{:.2f}%".format(rec))
            print('F1 Score : ', "{:.2f}%".format(f1))

        return scores