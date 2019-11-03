from PIL import Image

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import numpy as np
import os

# TODOs
# 1. Load all images and build feature vectors without the 'A' component in 'RGBA'
#    Try eliminating all the empty pixels (maybe Recursive Feature Elimination
#    will save our time).
# 2. Try applying logistic regression
# 3. Add cross-validation
# 4. Add SMOTE?
# 5. What are the ways to handle 'too many features but too few training samples'?

# References:
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# https://towardsdatascience.com/logistic-regression-for-facial-recognition-ab051acf6e4
# https://towardsdatascience.com/the-learning-process-logistic-regression-for-facial-recognition-take-2-6a1fef4ebe21
# https://scikit-learn.org/stable/modules/cross_validation.html

# Good read for extracting feature vectors from images, but probably not helpful for this project:
# https://www.analyticsvidhya.com/blog/2019/08/3-techniques-extract-features-from-image-data-machine-learning-python/

images_categories = []
images_data = []
images_names = []

direc = "./trainingSet/fire/"
for filename in os.listdir(direc):
    images_names.append(filename)
    image = Image.open(direc + filename)
    image = image.convert('RGB')
    images_categories.append(0)

    img_array = np.array(image)
    images_data.append(img_array)

direc = "./trainingSet/water/"
for filename in os.listdir(direc):
    images_names.append(filename)
    image = Image.open(direc + filename)
    image = image.convert('RGB')
    images_categories.append(1)

    img_array = np.array(image)
    images_data.append(img_array)

if not images_data:
    exit()

sample_image_data = images_data[0]
rows = len(sample_image_data)
columns = len(sample_image_data[0])

# hard-code ranges as all images have the same size
image_feature_required = [[[False for _ in range(3)] for _ in range(columns)] for _ in range(rows)]

#print(np.array(image_feature_required).shape)

num_features_required = 0
for image in images_data:
    for i in range(rows):
        for j in range(columns):
            for k in range(3):
                image_feature_required[i][j][k] = image_feature_required[i][j][k] or (image[i][j][k] != sample_image_data[i][j][k])

#print(np.sum(np.array(image_feature_required)))

image_features = []
for image in images_data:
    curr_image_feature = []
    for i in range(rows):
        for j in range(columns):
            for k in range(3):
                if image_feature_required[i][j][k]:
                    curr_image_feature.append(image[i][j][k])
    image_features.append(curr_image_feature)

image_features = np.asarray(image_features)

# now we have all the features, so we can start applying different techniques

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_features, images_categories, test_size=0.2, random_state=0)
print(y_test)

# Over-sampling (as Fire is under-sampled compare to Water)
# Random Over-sampliing
#ros = RandomOverSampler(random_state=0)
#X_train, y_train = ros.fit_resample(X_train, y_train)

# SMOTE
#os = SMOTE(random_state=0)
#X_train, y_train = os.fit_resample(X_train, y_train)

# fitting the model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# evaluate the model
# TODO: any other ways of measuring accuracy?
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
