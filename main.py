from PIL import Image
from numpy import asarray

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

# Good read for extracting feature vectors from images, but probably not helpful for this proj:
# https://www.analyticsvidhya.com/blog/2019/08/3-techniques-extract-features-from-image-data-machine-learning-python/

