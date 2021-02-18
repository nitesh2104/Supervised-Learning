import pydotplus
from IPython.display import Image
from io import StringIO
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import seaborn as sns
from helpers import read_file
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

class DT(object):
    def __init__(self, path, out_class):
        self.df = read_file(path)
        self.X = self.df.drop(columns=[out_class])
        self.y = self.df[[out_class]]
        self.clf = None

    def create_train_test_data(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size)
        return X_train, X_test, y_train, y_test

    def learner(self, X_train, y_train, **kwargs):
        """Create DecisionTreeClassifier"""
        if kwargs.get('dt_params'):
            self.clf = DecisionTreeClassifier(**kwargs['dt_params'])
        self.clf = DecisionTreeClassifier()
        self.clf = self.clf.fit(X_train, y_train)
        return self.clf

    def grid_search(self, parms):
        """Perform cv on dt using GridSearchCV"""
        clf = GridSearchCV(DecisionTreeClassifier(), parms, n_jobs=-1)
        clf.fit(X=self.X, y=self.y)
        tree_model = clf.best_estimator_
        return clf.best_score_, clf.best_params_

    def get_score(self, y_test, y_pred):
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print(f"Model accuracy {accuracy}")

    def get_confusion_matrix(self, y_test, y_pred):
        tp, fp, fn, tn = confusion_matrix(y_test, y_pred)
        a = confusion_matrix(y_test, y_pred)
        classification_report(y_test, y_pred)

    def cv_k_folds(self):
        kf = KFold(n_splits=3)
        kf.get_n_splits(self.X)

        for train_index, test_index in kf.split(self.X):
            print(f"Train Index: {train_index}, Test Index: {test_index}")

    def create_pipelines(self):
        """Create esimator pipelines"""
        pass

    def prune(self):
        """Prune most common occurring samples that are already decoded"""
        pass

    def plot_complexity_curve(self):
        """Plot the model complexity curve"""
        pass

    def cost_complexity_path(self):
        path = self.clf.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        return ccp_alphas, impurities

    def plot_graph(







            self, clf, save=False):
        feature_columns = (self.df.columns.tolist())
        feature_columns.remove('price_range')
        dot_data = StringIO()

        out_file = export_graphviz(clf,
                                   out_file=dot_data,
                                   filled=True,
                                   rounded=True,
                                   special_characters=True,
                                   feature_names=feature_columns,
                                   class_names=['0', '1', '2', '3']
                                   )
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        if save:
            graph.write_png('output.png')

        Image(graph.create_png())

    def plot_data(self, df, title="DT-Learner", xlabel="", ylabel="", legend=True):
        import matplotlib.pyplot as plt
        ax = df.plot(title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.show()


if __name__ == '__main__':
    path = "~/Projects/omscs/ML/train.csv"

    dt = DT(path, 'price_range')
    #sns.countplot(x="clock_speed", data=dt.df)
    #plt.show()
    X_train, X_test, y_train, y_test = dt.create_train_test_data(test_size=0.2)
    clf = dt.learner(X_train=X_train, y_train=y_train)
    y_pred = clf.predict(X_test)
    score_initial = dt.get_score(y_test, y_pred)
    print(score_initial)
    print("Here I am in this world making something of myself and and if ")
    # Understand the score better
    print(confusion_matrix(y_test,y_pred ))
    print(classification_report(y_test, y_pred))

    ### Plot the data against vanilla model set
    #dt.plot_graph(clf, save=True)

    params = {'max_depth': range(1, 8), 'criterion': ['gini', 'entropy']}
    clf_best_score, clf_best_params = dt.grid_search(params)
    print(clf_best_params)
    clf_cv = dt.learner(X_train=X_train, y_train=y_train, **clf_best_params)
    clf_cv.cv_results_
    y_pred_cv = clf_cv.predict(X_test)

    print(dt.get_score(y_test, y_pred_cv))
    print(confusion_matrix(y_test, y_pred_cv))
    print(classification_report(y_test, y_pred_cv))


    ## Use the new best params to plot the data
    #dt.plot_graph(clf, save=True)
    ## Plot data against the cv param model set and y_test
    ## Plot model complexity curve against the cv params set and y_test

    ## //TODO: Make sure to time the methods as well
