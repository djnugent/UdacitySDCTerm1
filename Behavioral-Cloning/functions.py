import os
import cv2
import numpy as np
from sklearn.utils import shuffle

# Preprocess data before it enters the network
def preprocess(img):
    #crop
    img = img[50:140]
    #blur
    img = cv2.GaussianBlur(img,(5,5),0)
    #YUV colorspace
    return cv2.cvtColor(img,cv2.COLOR_BGR2YUV)


# Generate batches of data
# Randomly samples
# Randomly drops near zero steering angles to make data more uniform
# Randomly selects from 3 cameras
# Randomly flips data
def gen(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)

        images = []
        measurements = []

        for sample in samples:
            #randomly drop near zero steering values
            measurement = float(sample[3])
            if(abs(measurement) < 0.1 and np.random.rand() < 0.7):
                continue

            #Randomly select left/right/center
            img_id = np.random.randint(3)
            filename = os.path.join('./IMG/',sample[img_id].split('\\')[-1])
            image = preprocess(cv2.imread(filename))
            if img_id == 1: #left
                measurement += 0.2
            elif img_id == 2: #right
                measurement -= 0.2

            #randomly flip sample
            flip = np.random.randint(2)
            if(flip):
                image = np.fliplr(image)
                measurement = -measurement

            images.append(image)
            measurements.append(measurement)

            #return batch
            if(len(images) >= batch_size):
                X = np.array(images)
                y = np.array(measurements)
                images = []
                measurements = []
                yield shuffle(X,y)

# Makes data more uniform by binning data based on steering angle
# Bins are trimmed down to average bin length
# I don't use this anymore
def uniform_data(samples,bin_num = 15):
    samples =np.array(samples)
    samples = shuffle(samples)

    #Extract steering angles
    angs = []
    for sample in samples:
        angs.append(float(sample[3]))

    #Bin data based on steering angle
    bin_lim = np.linspace(-1.1, 1.1, num=bin_num-1,endpoint=True)
    dig = np.digitize(angs,bin_lim,right=False)

    #place data in bins
    bins = []
    for i in range(bin_num):
        bins.append(samples[np.where(dig == i)])
    #Average bin length
    avg_len = 0
    for b in bins:
        avg_len += len(b)
    avg_len = int(avg_len/len(bins)/3)

    #trim all bins to average length
    for i,b in enumerate(bins):
        bins[i] = b[:avg_len]

    #merge trimmed bins back together
    uniform_samples = np.concatenate(bins)

    return uniform_samples
