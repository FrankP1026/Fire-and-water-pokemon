from PIL import Image

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

import argparse
import os

def add_images(img_directory, img_names, img_data, img_categories, category):
    for filename in os.listdir(img_directory):
        image = Image.open(img_directory + filename)
        image = image.convert('RGB')
        img_categories.append(category)

        img_array = np.array(image)
        img_names.append(filename)
        img_data.append(img_array)

def get_required_features(img_data):
    sample_image_data = img_data[0]
    rows = len(sample_image_data)
    columns = len(sample_image_data[0])

    # hard-code ranges as all images have the same size
    image_feature_required = [[[False for _ in range(3)] for _ in range(columns)] for _ in range(rows)]

    #print(np.array(image_feature_required).shape)

    for image in img_data:
        for i in range(rows):
            for j in range(columns):
                for k in range(3):
                    image_feature_required[i][j][k] = image_feature_required[i][j][k] or (image[i][j][k] != sample_image_data[i][j][k])
    
    return image_feature_required

def get_flattened_features(img_data, img_feature_required):
    image_features = []
    for image in img_data:
        curr_image_feature = []
        for i in range(rows):
            for j in range(columns):
                for k in range(3):
                    if img_feature_required[i][j][k]:
                        curr_image_feature.append(image[i][j][k] / 255)
        image_features.append(curr_image_feature)

    return np.asarray(image_features)

# Steps
# 1. Load all images and build feature vectors without the 'A' component in 'RGBA'
#    Try eliminating all the empty pixels (maybe Recursive Feature Elimination
#    will save our time).
# 2. Try Principal Component Analysis (PCA) / Recursive Feature Elimination
# 3. Try Oversampling
# 4. Try applying an algorithm (Logistic Regression, SVM, Neural Network, etc.)
# 5. Do cross-validation

# References:
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# https://towardsdatascience.com/logistic-regression-for-facial-recognition-ab051acf6e4
# https://towardsdatascience.com/the-learning-process-logistic-regression-for-facial-recognition-take-2-6a1fef4ebe21
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://towardsdatascience.com/feature-selection-in-python-recursive-feature-elimination-19f1c39b8d15

# Good read for extracting feature vectors from images, but probably not helpful for this project:
# https://www.analyticsvidhya.com/blog/2019/08/3-techniques-extract-features-from-image-data-machine-learning-python/

if __name__ == '__main__':
    # Whether Principal Component Analysis is used for training or not
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_pca', dest='use_pca', action='store_true')
    parser.set_defaults(use_pca=False)
    args = parser.parse_args()
    use_pca = args.use_pca

    images_categories = []
    images_data = []
    images_names = []

    add_images("./trainingSet/fire/", images_names, images_data, images_categories, 0)
    add_images("./trainingSet/water/", images_names, images_data, images_categories, 1)

    if not images_data:
        exit()

    sample_image_data = images_data[0]
    rows = len(sample_image_data)
    columns = len(sample_image_data[0])

    image_feature_required = get_required_features(images_data)
    #print(np.sum(np.array(image_feature_required)))

    # flatten the features
    image_features = get_flattened_features(images_data, image_feature_required)

    # Cross Validation

    # TODO: do over-sampling on the training data for each iteration of cross validation.
    # See https://datascience.stackexchange.com/questions/45046/cross-validation-for-highly-imbalanced-data-with-undersampling
    # and https://www.researchgate.net/post/should_oversampling_be_done_before_or_within_cross-validation

    # Cross Validation is NOT used for fitting the model, it is for testing the performance of proposed models,
    # fit() has to be called at the end (with all training data) to finalize the model
    # See https://stats.stackexchange.com/questions/52274/how-to-choose-a-predictive-model-after-k-fold-cross-validation
    #
    # TODO: try more advanced cross-validation data-spliting methods, like StratifiedKFold
    logreg_lasso = LogisticRegression(solver='liblinear', penalty="l1")
    scores = cross_val_score(logreg_lasso, image_features, images_categories, cv=5)
    print(scores)
    print("Accuracy of cross_validation with Lasso regularization: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    logreg_ridge = LogisticRegression(solver='lbfgs', penalty="l2")
    scores = cross_val_score(logreg_ridge, image_features, images_categories, cv=5)
    print(scores)
    print("Accuracy of cross_validation with Ridge regularization: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # fitting the models
    X_train, X_test, y_train, y_test = train_test_split(image_features, images_categories, test_size=0.2)

    # Using original dataset for training does not help with eventual performance
    #X_train = image_features
    #y_train = images_categories

    # Over-sampling (as Fire is under-sampled compare to Water)
    # Random Over-sampliing
    #ros = RandomOverSampler()
    #X_train, y_train = ros.fit_resample(X_train, y_train)

    # SMOTE
    #smote_os = SMOTE()
    #X_train, y_train = smote_os.fit_resample(X_train, y_train)

    pca = PCA(len(X_train))
    if use_pca:
        # PCA for Visulization
        pca = PCA(2)  # project to 2 dimensions
        projected = pca.fit_transform(X_train)
        colours = ['r', 'b']
        for i in range(len(colours)):
            x_values = [x for idx, x in enumerate(projected[:, 0]) if y_train[idx] == i]
            y_values = [y for idx, y in enumerate(projected[:, 1]) if y_train[idx] == i]
            plt.scatter(x_values, y_values, c=colours[i], edgecolor='none', alpha=0.5)
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.show()

        # PCA for explained variances and selecting number of principal components
        pca = PCA().fit(X_train)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()

        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    logreg_lasso.fit(X_train, y_train)
    y_pred = logreg_lasso.predict(X_test)
    print('Accuracy of logistic regression classifier with Lasso regularization on test set extracted by \
    train_test_split(): {:.2f}'.format(logreg_lasso.score(X_test, y_test)))

    logreg_ridge.fit(X_train, y_train)
    y_pred = logreg_ridge.predict(X_test)
    print('Accuracy of logistic regression classifier with Ridge regularization on test set extracted by \
    train_test_split(): {:.2f}'.format(logreg_ridge.score(X_test, y_test)))

    # testing code
    images_categories = []
    images_data = []
    images_names = []
    image_features = []

    add_images("./test_set_water_and_fire/Fire/", images_names, images_data, images_categories, 0)
    add_images("./test_set_water_and_fire/Water/", images_names, images_data, images_categories, 1)
    if not images_data:
        exit()

    image_features = get_flattened_features(images_data, image_feature_required)
    if use_pca:
        image_features = pca.transform(image_features)

    y_pred = logreg_lasso.predict(image_features)
    print('Accuracy of logistic regression classifier with Lasso Regularization on actual test set \
    : {:.2f}'.format(logreg_lasso.score(image_features, images_categories)))

    y_pred = logreg_ridge.predict(image_features)
    print('Accuracy of logistic regression classifier with Ridge Regularization on actual test set \
    : {:.2f}'.format(logreg_ridge.score(image_features, images_categories)))
