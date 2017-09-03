import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from lesson_functions import extract_feature, imread, convert_color
from sklearn.externals import joblib

try:
    # Read in car and non-car images
    images = glob.iglob('../dataset/**/*.png', recursive=True)
    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        elif 'vehicles' in image:
            cars.append(image)

    #cars = cars[:100]
    #notcars = notcars[:100]

    # Extract features
    spatial_size = (16,16)
    hist_bins = 32
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    t = time.time()
    car_features = []
    for img_name in cars:
        img = imread(img_name)
        img = convert_color(img,'HLS')
        feature = extract_feature(img,orient=orient,
                            pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            spatial_size=spatial_size,
                            hist_bins=hist_bins)
        car_features.append(feature)
    print(time.time() - t, "Seconds to extract car features")

    t = time.time()
    notcar_features = []
    for img_name in notcars:
        img = imread(img_name)
        img = convert_color(img,'HLS')
        feature = extract_feature(img,orient=orient,
                            pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            spatial_size=spatial_size,
                            hist_bins=hist_bins)
        notcar_features.append(feature)
    print(time.time() - t, "Seconds to extract non car features")

    print(len(car_features),len(notcar_features))

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    joblib.dump(X_scaler, 'scaler.pkl')

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.ones(len(notcar_features))*-1))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using spatial binning of:',spatial_size,
        'and', hist_bins,'histogram bins')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()

    # Optimize parameters
    parameters = { 'C': [1, 10,100]}
    clf = GridSearchCV(svc, parameters,verbose=3)

    # Check the training time for the SVC
    t=time.time()
    clf.fit(X_train, y_train)
    svc = clf.best_estimator_
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    print("Saving model")
    joblib.dump(svc, 'svm.pkl')


except Exception as e:
    print(e)
    import IPython
    IPython.embed()
