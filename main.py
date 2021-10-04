import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt
import os
from sklearn.svm import LinearSVC
import math


def preprocessing(image):
    img = copy.deepcopy(image)
    N, M, D = img.shape
    resized_img = cv2.resize(img, (64, 128))
    return resized_img, N, M


def gradient(image):
    img = copy.deepcopy(image)
    img = np.float32(img) / 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    return gx, gy, mag, angle


def value_correction(mag):
    N, M, D = mag.shape
    for i in range(0, N):
        for j in range(0, M):
            mag[i, j, 0] = round(mag[i, j, 0], 3)
            mag[i, j, 1] = round(mag[i, j, 1], 3)
            mag[i, j, 2] = round(mag[i, j, 2], 3)
    mag = np.uint8(mag * 255)
    return mag


def visualize_gradient(img):
    viz_img = copy.deepcopy(img)
    resized_image = cv2.resize(viz_img, (64, 128))
    resized_image = np.float32(resized_image) / 255.0
    gx_copy = cv2.Sobel(resized_image, cv2.CV_32F, 1, 0, ksize=1)
    gy_copy = cv2.Sobel(resized_image, cv2.CV_32F, 0, 1, ksize=1)
    mag_copy, angle_copy = cv2.cartToPolar(gx_copy, gy_copy, angleInDegrees=True)
    gx_copy = cv2.resize(gx_copy, (200, 400))
    gy_copy = cv2.resize(gy_copy, (200, 400))
    mag_copy = cv2.resize(mag_copy, (200, 400))
    img_show = cv2.hconcat([abs(gx_copy), abs(gy_copy), mag_copy])
    cv2.imshow('test', img_show)
    cv2.waitKey(0)


def visualize_bins(img):
    img_hist = copy.deepcopy(img)
    M, N, D = img_hist.shape
    for i in range(7, N - 8, 8):
        cv2.line(img_hist, (i, 0), (i, M), (255, 0, 0), 1)
    for j in range(7, M - 8, 8):
        cv2.line(img_hist, (0, j), (N, j), (255, 0, 0), 1)
    img_hist = cv2.resize(img_hist, (200, 400))
    cv2.imshow('test', img_hist)
    cv2.waitKey(0)


def round_numbers(mag, angle):
    m_R = mag[:, :, 0]
    m_G = mag[:, :, 1]
    m_B = mag[:, :, 2]
    a_R = angle[:, :, 0]
    a_G = angle[:, :, 1]
    a_B = angle[:, :, 2]

    N, M, D = mag.shape
    mag_RGB = np.zeros((N, M))
    angle_RGB = np.zeros((N, M))
    for i in range(0, N):
        for j in range(0, M):
            vec = [m_R[i, j], m_G[i, j], m_B[i, j]]
            max_value = np.max(vec)
            if max_value == vec[0]:
                mag_RGB[i, j] = round(max_value)
                angle_RGB[i, j] = round(a_R[i, j])
            elif max_value == vec[1]:
                mag_RGB[i, j] = round(max_value)
                angle_RGB[i, j] = round(a_G[i, j])
            elif max_value == vec[2]:
                mag_RGB[i, j] = round(max_value)
                angle_RGB[i, j] = round(a_B[i, j])
    return mag_RGB, angle_RGB


def divide_blocks(mag, angle):
    N_mag, M_mag = mag.shape
    magnitude_RGB = [[], [], [], [], [], [], [], []]
    angle_RGB = [[], [], [], [], [], [], [], []]
    step = 8
    counter = 0
    for i in range(0, N_mag, 8):
        for j in range(0, M_mag, 8):
            mag_RGB = mag[i:i + step, j:j + step]
            ang_RGB = angle[i:i + step, j:j + step]
            magnitude_RGB[counter].append(mag_RGB)
            angle_RGB[counter].append(ang_RGB)

            if len(angle_RGB[counter]) == 16:
                counter = counter + 1
    return magnitude_RGB, angle_RGB


def histogram_proportions(interval, an, ma, bin1, bin2):
    if len(interval) == 20:
        if an == interval[0]:
            bin1.append(ma)
        elif an == interval[1]:
            bin1.append(round(ma * 0.95))
            bin2.append(round(ma * 0.05))
        elif an == interval[2]:
            bin1.append(round(ma * 0.9))
            bin2.append(round(ma * 0.1))
        elif an == interval[3]:
            bin1.append(round(ma * 0.85))
            bin2.append(round(ma * 0.15))
        elif an == interval[4]:
            bin1.append(round(ma * 0.8))
            bin2.append(round(ma * 0.2))
        elif an == interval[5]:
            bin1.append(round(ma * 0.75))
            bin2.append(round(ma * 0.25))
        elif an == interval[6]:
            bin1.append(round(ma * 0.7))
            bin2.append(round(ma * 0.3))
        elif an == interval[7]:
            bin1.append(round(ma * 0.65))
            bin2.append(round(ma * 0.35))
        elif an == interval[8]:
            bin1.append(round(ma * 0.6))
            bin2.append(round(ma * 0.4))
        elif an == interval[9]:
            bin1.append(round(ma * 0.55))
            bin2.append(round(ma * 0.45))
        elif an == interval[10]:
            bin1.append(round(ma / 2))
            bin2.append(round(ma / 2))
        elif an == interval[11]:
            bin1.append(round(ma * 0.45))
            bin2.append(round(ma * 0.55))
        elif an == interval[12]:
            bin1.append(round(ma * 0.4))
            bin2.append(round(ma * 0.6))
        elif an == interval[13]:
            bin1.append(round(ma * 0.35))
            bin2.append(round(ma * 0.65))
        elif an == interval[14]:
            bin1.append(round(ma * 0.3))
            bin2.append(round(ma * 0.7))
        elif an == interval[15]:
            bin1.append(round(ma * 0.25))
            bin2.append(round(ma * 0.75))
        elif an == interval[16]:
            bin1.append(round(ma * 0.2))
            bin2.append(round(ma * 0.8))
        elif an == interval[17]:
            bin1.append(round(ma * 0.15))
            bin2.append(round(ma * 0.85))
        elif an == interval[18]:
            bin1.append(round(ma * 0.1))
            bin2.append(round(ma * 0.9))
        elif an == interval[19]:
            bin1.append(round(ma * 0.05))
            bin2.append(round(ma * 0.95))
    elif len(interval) == 21:
        if an == interval[0]:
            bin1.append(ma)
        elif an == interval[1]:
            bin1.append(round(ma * 0.95))
            bin2.append(round(ma * 0.05))
        elif an == interval[2]:
            bin1.append(round(ma * 0.9))
            bin2.append(round(ma * 0.1))
        elif an == interval[3]:
            bin1.append(round(ma * 0.85))
            bin2.append(round(ma * 0.15))
        elif an == interval[4]:
            bin1.append(round(ma * 0.8))
            bin2.append(round(ma * 0.2))
        elif an == interval[5]:
            bin1.append(round(ma * 0.75))
            bin2.append(round(ma * 0.25))
        elif an == interval[6]:
            bin1.append(round(ma * 0.7))
            bin2.append(round(ma * 0.3))
        elif an == interval[7]:
            bin1.append(round(ma * 0.65))
            bin2.append(round(ma * 0.35))
        elif an == interval[8]:
            bin1.append(round(ma * 0.6))
            bin2.append(round(ma * 0.4))
        elif an == interval[9]:
            bin1.append(round(ma * 0.55))
            bin2.append(round(ma * 0.45))
        elif an == interval[10]:
            bin1.append(round(ma / 2))
            bin2.append(round(ma / 2))
        elif an == interval[11]:
            bin1.append(round(ma * 0.45))
            bin2.append(round(ma * 0.55))
        elif an == interval[12]:
            bin1.append(round(ma * 0.4))
            bin2.append(round(ma * 0.6))
        elif an == interval[13]:
            bin1.append(round(ma * 0.35))
            bin2.append(round(ma * 0.65))
        elif an == interval[14]:
            bin1.append(round(ma * 0.3))
            bin2.append(round(ma * 0.7))
        elif an == interval[15]:
            bin1.append(round(ma * 0.25))
            bin2.append(round(ma * 0.75))
        elif an == interval[16]:
            bin1.append(round(ma * 0.2))
            bin2.append(round(ma * 0.8))
        elif an == interval[17]:
            bin1.append(round(ma * 0.15))
            bin2.append(round(ma * 0.85))
        elif an == interval[18]:
            bin1.append(round(ma * 0.1))
            bin2.append(round(ma * 0.9))
        elif an == interval[19]:
            bin1.append(round(ma * 0.05))
            bin2.append(round(ma * 0.95))
        elif an == interval[20]:
            bin2.append(ma)

    return bin1, bin2


def histogram_of_gradients(magnitude, angle):
    interval1 = list(range(0, 20))
    interval2 = list(range(20, 40))
    interval3 = list(range(40, 60))
    interval4 = list(range(60, 80))
    interval5 = list(range(80, 100))
    interval6 = list(range(100, 120))
    interval7 = list(range(120, 140))
    interval8 = list(range(140, 160))
    interval9 = list(range(160, 181))

    histogram = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    histogram_visualization = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    counter2 = 0
    for a in range(0, len(magnitude)):
        for b in range(0, len(magnitude[0])):
            ma = magnitude[a][b]
            an = angle[a][b]
            bin0 = []
            bin20 = []
            bin40 = []
            bin60 = []
            bin80 = []
            bin100 = []
            bin120 = []
            bin140 = []
            bin160 = []

            for i in range(0, 8):
                for j in range(0, 8):
                    if an[i, j] > 180:
                        an[i, j] = an[i, j] - 180
                    if an[i, j] in interval1:
                        bin0, bin20 = histogram_proportions(interval1, an[i, j], ma[i, j], bin0, bin20)
                    elif an[i, j] in interval2:
                        bin20, bin40 = histogram_proportions(interval2, an[i, j], ma[i, j], bin20, bin40)
                    elif an[i, j] in interval3:
                        bin40, bin60 = histogram_proportions(interval3, an[i, j], ma[i, j], bin40, bin60)
                    elif an[i, j] in interval4:
                        bin60, bin80 = histogram_proportions(interval4, an[i, j], ma[i, j], bin60, bin80)
                    elif an[i, j] in interval5:
                        bin80, bin100 = histogram_proportions(interval5, an[i, j], ma[i, j], bin80, bin100)
                    elif an[i, j] in interval6:
                        bin100, bin120 = histogram_proportions(interval6, an[i, j], ma[i, j], bin100, bin120)
                    elif an[i, j] in interval7:
                        bin120, bin140 = histogram_proportions(interval7, an[i, j], ma[i, j], bin120, bin140)
                    elif an[i, j] in interval8:
                        bin140, bin160 = histogram_proportions(interval8, an[i, j], ma[i, j], bin140, bin160)
                    elif an[i, j] in interval9:
                        bin160, bin0 = histogram_proportions(interval9, an[i, j], ma[i, j], bin160, bin0)

            # hist = [len(bin0), len(bin20), len(bin40), len(bin60), len(bin80), len(bin100), len(bin120), len(bin140),
            #         len(bin160)]

            hist = [(np.sum(bin0)), (np.sum(bin20)), (np.sum(bin40)), (np.sum(bin60)), (np.sum(bin80)),
                    (np.sum(bin100)), (np.sum(bin120)), (np.sum(bin140)), (np.sum(bin160))]

            hist_visualization = [bin0, bin20, bin40, bin60, bin80, bin100, bin120, bin140, bin160]
            histogram[counter2].append(hist)
            histogram_visualization[counter2].append(hist_visualization)

            if len(histogram[counter2]) == 8:
                counter2 = counter2 + 1
    return histogram, histogram_visualization


def visualize_histogram(histogram, i, j):
    hist = histogram[i][j]
    for i in range(0, len(hist)):
        hist[i] = hist[i] * 100
    number_bins_hist = [(hist[0]), (hist[1]), (hist[2]),
                        (hist[3]), (hist[4]), (hist[5]),
                        (hist[6]), (hist[7]), (hist[8])]
    N = 9
    ind = np.arange(N)
    width = 0.50
    plt.bar(ind, number_bins_hist, width, color='red')
    plt.title('Przykładowy histogram')
    plt.xlabel('Kierunek')
    plt.ylabel('Zliczenia')
    plt.grid(True)
    plt.xticks(ind, ('0', '20', '40', '60', '80', '100', '120', '140', '160'))
    plt.show()


def histogram_normalization(histogram):
    FeatureDescriptor = []
    norm_hist = []
    histogram_image = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for a in range(0, len(histogram_image)):
        for b in range(0, 8):
            histogram_image[a].append([])

    for i in range(0, len(histogram) - 1):
        for j in range(0, len(histogram[0]) - 1):
            tmp = [histogram[i][j], histogram[i][j + 1], histogram[i + 1][j], histogram[i + 1][j + 1]]
            value = 0
            for k in range(0, len(tmp)):
                for z in range(0, 9):
                    value = value + pow(tmp[k][z], 2)
            sqrt_value = round(np.sqrt(value), 2)
            for a in range(0, len(tmp)):
                norm_hist = []
                for b in range(0, 9):
                    norm_value = round((tmp[a][b] / sqrt_value), 2)
                    norm_hist.append(norm_value)
                    FeatureDescriptor.append(norm_value)

                    if (i % 2) == 0 and (j % 2) == 0:
                        if a == 0:
                            histogram_image[i][j].append(norm_value)
                        elif a == 1:
                            histogram_image[i][j + 1].append(norm_value)
                        elif a == 2:
                            histogram_image[i + 1][j].append(norm_value)
                        elif a == 3:
                            histogram_image[i + 1][j + 1].append(norm_value)
    return histogram_image, FeatureDescriptor


def visualization_function(matrix, length, a):
    a = math.radians(a)
    y_start = 100
    x_start = 100
    y_end = int(y_start + (length * np.sin(a)))
    x_end = int(x_start + (length * np.cos(a)))
    y_end_mirror = int(y_start - (y_end - y_start))
    x_end_mirror = int(x_start - (x_end - x_start))
    cv2.line(matrix, (x_start, y_start), (x_end, y_end), (0, 0, 255), 1)
    cv2.line(matrix, (x_start, y_start), (x_end_mirror, y_end_mirror), (0, 0, 255), 1)
    return matrix


def visualization_vector(histogram):
    viz = np.zeros((200, 200))
    viz = (np.dstack((viz, viz, viz)) * 255.999).astype(np.uint8)
    angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    for i in range(0, len(histogram)):
        length = histogram[i] * 100
        a = angles[i]
        viz = visualization_function(viz, length, a)
    viz = cv2.transpose(viz)
    viz = cv2.flip(viz, flipCode=0)
    return viz


def image_vis(histogram):
    viz_vertical = []
    for i in range(0, len(histogram)):
        viz_horizontal = []
        for j in range(0, len(histogram[0])):
            v = visualization_vector(histogram[i][j])
            v = cv2.resize(v, (300, 300))
            viz_horizontal.append(v)
        viz_vertical.append(viz_horizontal)

    vertical_image = []
    for i in range(0, len(viz_vertical)):
        horizontal_image = []
        for j in range(0, len(viz_vertical[0])):
            horizontal_image.append(viz_vertical[i][j])
        horizontal_image = cv2.hconcat(
            [horizontal_image[0], horizontal_image[1], horizontal_image[2], horizontal_image[3],
             horizontal_image[4], horizontal_image[5], horizontal_image[6], horizontal_image[7]])
        vertical_image.append(horizontal_image)

    image_visualization = cv2.vconcat(
        [vertical_image[0], vertical_image[1], vertical_image[2], vertical_image[3], vertical_image[4],
         vertical_image[5], vertical_image[6], vertical_image[7], vertical_image[8], vertical_image[9],
         vertical_image[10], vertical_image[11], vertical_image[12], vertical_image[13], vertical_image[14],
         vertical_image[15]])

    return image_visualization


def HOG_visualization(original_image, vector_image, N, M):
    image_visualization = vector_image + vector_image + vector_image
    image_visualization = cv2.resize(image_visualization, (3*M, 3*N))
    img_rez = cv2.resize(original_image, (3*M, 3*N))
    M, N, D = img_rez.shape
    for i in range(0, M):
        for j in range(0, N):
            if image_visualization[i, j, 2] != 0:
                img_rez[i, j, 0] = 0
                img_rez[i, j, 1] = 0
                img_rez[i, j, 2] = 255
    return img_rez, image_visualization


def HOG(img):
    resized_img, N, M = preprocessing(img)
    gx, gy, mag, angle = gradient(resized_img)
    mag = value_correction(mag)
    mag, angle = round_numbers(mag, angle)
    # visualize_gradient(img)
    # visualize_bins(resized_img)
    magnitude_RGB, angle_RGB = divide_blocks(mag, angle)
    histogram, histogram_visualization = histogram_of_gradients(magnitude_RGB, angle_RGB)
    # visualize_histogram(histogram, 1, 3)
    histogram_image, FeatureDescriptor = histogram_normalization(histogram)
    image_visualization = image_vis(histogram_image)
    HOG_image, image_vectors = HOG_visualization(img, image_visualization, N, M)
    return HOG_image, image_vectors, FeatureDescriptor


def load_images_from_folder(folder):
    images = []
    names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        names.append(filename)
        if img is not None:
            images.append(img)
    return images, names


def predicionSVM():
    folder = "C:/Users/Admin/PycharmProjects/HoG/input"
    SVM_input, names = load_images_from_folder(folder)
    images = []
    labels = ['1', '1', '1', '1', '1', '1', '0', '0', '0', '0']
    for i in range(0, len(SVM_input)):
        hog_img, image_vectors, hog_desc = HOG(SVM_input[i])
        images.append(hog_desc)
        # labels.append(names[i])
    print('Training on train images...')
    svm_model = LinearSVC(random_state=42, tol=1e-5)
    svm_model.fit(images, labels)

    folder2 = "C:/Users/Admin/PycharmProjects/HoG/test_images"
    test, labelz = load_images_from_folder(folder2)
    print('Evaluating on test images...')
    for j in range(0, len(test)):
        hog_image, image_vectors, hog_desc_test = HOG(test[j])
        hog_desc_test = np.array(hog_desc_test)
        pred = svm_model.predict(hog_desc_test.reshape(1, -1))[0]
        cv2.imshow('HOG Image', image_vectors)
        cv2.waitKey(0)
        cv2.imshow('HOG Image', hog_image)
        cv2.waitKey(0)
        N, M, D = test[j].shape
        test[j] = cv2.resize(test[j], (3*M, 3*N))
        cv2.putText(test[j], pred.title(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 2)
        cv2.imshow('Test Image', test[j])
        cv2.waitKey(0)


def main():
    img = cv2.imread("runner.png")
    HOG_image, image_vectors, FeatureDescriptor = HOG(img)
    cv2.imshow('HOG image', HOG_image)
    cv2.waitKey(0)
    # predicionSVM()


main()
