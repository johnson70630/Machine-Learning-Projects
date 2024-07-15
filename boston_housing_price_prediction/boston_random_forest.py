"""
This file demonstrates how to analyze boston
housing dataset. Students will upload their
results to kaggle.com and compete with people
in class!
"""

import pandas as pd
from sklearn import preprocessing, ensemble, metrics, model_selection

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'


def main():
    # data preprocessing
    data = pd.read_csv(TRAIN_FILE)
    train_data, val_data = model_selection.train_test_split(data, test_size=0.4)
    X_train, Y = data_preprocess(train_data, mode='Train')
    X_train = one_hot_encoding(X_train)

    # training
    forest = ensemble.RandomForestRegressor(min_samples_leaf=6)
    bagging = ensemble.BaggingRegressor(base_estimator=forest)
    poly_phi = preprocessing.PolynomialFeatures(degree=2)
    X_train_poly = poly_phi.fit_transform(X_train)
    regressor_poly_bagging = bagging.fit(X_train_poly, Y)

    # validating
    X_val, Y = data_preprocess(val_data, mode='Train')
    X_val = one_hot_encoding(X_val)
    X_val_poly = poly_phi.transform(X_val)
    validation = regressor_poly_bagging.predict(X_val_poly)
    print('Error: ', metrics.mean_squared_error(validation, Y) ** 0.5)

    # predicting
    data = pd.read_csv(TEST_FILE)
    X_test, id = data_preprocess(data, mode='Test')
    X_test = one_hot_encoding(X_test)
    X_test_poly = poly_phi.transform(X_test)
    predictions = regressor_poly_bagging.predict(X_test_poly)
    print('predictions:', predictions)
    out_file(predictions, id, 'boston_housing_random_forest_bagging.csv')


def data_preprocess(data, mode='Train'):
    id = data.pop('ID')
    # data.loc(data['zn'] <= data['zn'].quantile(q=0.25), 'zn') = 1
    # data.loc(data['zn'].quantile(q=0.25) < data.zn <= data['zn'].quantile(q=0.5), 2)
    # data.loc(data['zn'].quantile(q=0.5) < data.zn <= data['zn'].quantile(q=0.75), 3)
    # data.loc(data.zn > data['zn'].quantile(q=0.75), 4)
    if mode == 'Train':
        labels = data.pop('medv')
        return data, labels
    else:
        return data, id


def one_hot_encoding(data):
    data = pd.get_dummies(data, columns=['chas'])
    return data


def out_file(predictions, id, filename):
    print('\n===============================================')
    print(f'Writing predictions to --> {filename}')
    with open(filename, 'w') as out:
        out.write('ID,medv\n')
        for i in range(len(id)):
            out.write(str(id[i]) + ',' + str(predictions[i]) + '\n')
    print('===============================================')


def standardization(data, mode='Train'):
    standardizer = preprocessing.StandardScaler()
    if mode == 'Train':
        data = standardizer.fit_transform(data)
    else:
        data = standardizer.transform(data)
    return data


if __name__ == '__main__':
    main()
