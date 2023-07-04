import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from src.earthquake import Earthquake
from src.constants import X_LABEL, Y_LABEL, TITLE, FILENAME

warnings.filterwarnings('ignore')

mse, r2 = 'mse', 'r^2'
scores = {
    "Model name": ["Linear regression", "SVM", "Random Forest"],
    mse: [],
    r2: []
}
file_name = './dataset/Earthquake_data_processed.xlsx'

def preproces():
    df = pd.read_csv('./dataset/Earthquake_Data.csv', delimiter=r'\s+')
    new_column_names = [
        "Date(YYYY/MM/DD)",  "Time(UTC)", "Latitude(deg)", "Longitude(deg)", "Depth(km)", "Magnitude(ergs)", 
        "Magnitude_type", "No_of_Stations", "Gap", "Close", "RMS", "SRC", "EventID"
    ]

    df.columns = new_column_names
    ts = pd.to_datetime(df["Date(YYYY/MM/DD)"] + " " + df["Time(UTC)"])
    df = df.drop(["Date(YYYY/MM/DD)", "Time(UTC)"], axis=1)
    df.index = ts
    df.to_excel(file_name)

def get_dataset(input_features, output_feature):
    df = pd.read_excel(file_name)
    X = df[input_features]
    y = df[output_feature]
    return X, y

def put_scores(earthquake):
    r2_score, mse_score = earthquake.get_scores()
    print("R^2: {:.2f}, MSE: {:.2f}".format(r2_score, mse_score))
    scores[mse].append(mse_score)
    scores[r2].append(r2_score)

def linear_regression():
    print('Linear Regression')
    input_features = ['Latitude(deg)', 'Longitude(deg)', 'Depth(km)', 'No_of_Stations']
    X, y = get_dataset(input_features, 'Magnitude(ergs)')
    earthquake = Earthquake(X, y, LinearRegression())
    earthquake.split(0.2)
    earthquake.train()
    earthquake.test()
    put_scores(earthquake)
    earthquake.plot_predictions({
        X_LABEL: 'Actual Magnitude',
        Y_LABEL: 'Predicted Maginitute',
        TITLE: 'Linear Regression',
        FILENAME: './plots/linear_regression_predict.png'
    })
    earthquake.feature_plot(input_features, {
        X_LABEL: 'Actual Magnitude',
        Y_LABEL: 'Predictor Variables',
        TITLE: 'Linear Regression',
        FILENAME: './plots/linear_regression_features.png'
    })

def support_vector_machine():
    print('Support Vector Machine')
    input_features = ['Latitude(deg)', 'Longitude(deg)', 'Depth(km)', 'No_of_Stations']
    X, y = get_dataset(input_features, 'Magnitude(ergs)')
    earthquake = Earthquake(X, y, SVR(kernel='rbf', C=1e3, gamma=0.1))
    earthquake.split(0.2)
    earthquake.train(subset_size = 500)
    earthquake.test()
    put_scores(earthquake)
    earthquake.plot_predictions({
        X_LABEL: 'Actual Magnitude',
        Y_LABEL: 'Predicted Maginitute',
        TITLE: 'SVM',
        FILENAME: './plots/svm_predict.png'
    })
    earthquake.feature_plot(input_features, {
        X_LABEL: 'Actual Magnitude',
        Y_LABEL: 'Predictor Variables',
        TITLE: 'SVM',
        FILENAME: './plots/svm_features.png'
    })

if __name__ == '__main__':
    preproces()
    linear_regression()
    support_vector_machine()