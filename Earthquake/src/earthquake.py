import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from .constants import RANDOM_STATE, X_LABEL, Y_LABEL, TITLE, FILENAME

colors = ['blue', 'red', 'yellow', 'violet', 'orange', 'black']

class Earthquake:
    def __init__(self, X, y, model):
        self.X, self.y = X, y
        self.model = model

    def split(self, test_size):
        data_split = train_test_split(self.X, self.y, test_size=test_size, random_state=RANDOM_STATE)
        self.X_train, self.X_test, self.y_train, self.y_test = data_split

    def train(self, subset_size = None):
        if subset_size is None:
            subset_size = self.X_train.shape[0]
        self.model.fit(self.X_train[:subset_size], self.y_train[:subset_size])

    def test(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_scores(self):
        return r2_score(self.y_test, self.y_pred), mean_squared_error(self.y_test, self.y_pred)
    
    def configure_plot(self, plot_config):
        plt.xlabel(plot_config[X_LABEL])
        plt.ylabel(plot_config[Y_LABEL])
        plt.title(plot_config[TITLE])
        plt.savefig(plot_config[FILENAME], format='png')
        plt.close()
    
    def plot_predictions(self, plot_config):
        plt.scatter(self.y_test, self.y_pred)
        self.configure_plot(plot_config)

    def feature_plot(self, features, plot_config):
        for i, feature in enumerate(features):
            sns.regplot(x = self.X_test[feature], y = self.y_test, color = colors[i], scatter_kws={'s': 10})
        plt.legend(labels = features)
        self.configure_plot(plot_config)

