import os

import matplotlib.pyplot as plt
# for everything else
import numpy as np
from keras.applications.vgg16 import preprocess_input
# clustering and dimension reduction
from sklearn.cluster import KMeans
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img


def extract_features(file, model, train_image_directory):
    # load the image as a 224x224 array
    print(train_image_directory + file)
    img = load_img(os.path.join(train_image_directory, file), target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


def get_folder_and_extract_features(train_image_directory, folder_name, model_file, model_weights):
    # this list holds all the image filename
    trailer_1 = []
    # filenames=[]
    print(train_image_directory)
    # creates a ScandirIterator aliased as files
    with os.scandir(train_image_directory) as files:
        # loops through each file in the directory
        for file in files:
            if file.name.endswith('.jpg'):
                # adds only the image files to the flowers list
                # print(os.path.join(os.getcwd(), file))
                trailer_1.append(file.name)
                # filenames.append(file.name)
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights)
    print("Loaded model from disk")
    # print(trailer_1)
    data = {}
    for index, images in enumerate(trailer_1):
        # try to extract the features and update the dictionary
        feat = extract_features(images, loaded_model, train_image_directory)
        data[images] = feat
        print("Image", index, "predicted")

    filenames = np.array(list(data.keys()))

    # get a list of just the features
    feat = np.array(list(data.values()))
    print(feat.shape)
    # (210, 1, 4096)

    # reshape so that there are 210 samples of 4096 vectors
    feat = feat.reshape(-1, 4096)
    print(feat.shape)

    # reduce the amount of dimensions in the feature vector
    # pca = PCA(n_components=80, random_state=22)
    # pca.fit(feat)
    # x = pca.transform(feat)

    # cluster feature vectors
    kmeans = KMeans(n_clusters=5, random_state=22)
    kmeans.fit(feat)

    # holds the cluster id and the images { id: [images] }
    groups = {}
    for file, cluster in zip(filenames, kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    print(groups)

    # function that lets you view a cluster (based on identifier)
    def view_cluster(cluster, train_image_directory, folder_name):
        plt.figure(figsize=(25, 25));
        # gets the list of filenames for a cluster
        files = groups[cluster]
        # only allow up to 30 images to be shown at a time
        # if len(files) > 30:
        #     print(f"Clipping cluster size from {len(files)} to 30")
        #     files = files[:29]
        # plot each image in the cluster
        for index, file in enumerate(files):
            plt.subplot(10, 10, index + 1);
            img = load_img(os.path.join(train_image_directory, file))
            img = np.array(img)
            plt.imshow(img)

            plt.axis('off')
        plt.savefig(folder_name + '_c' + str(cluster) + '.png')
        # plt.show()

    print('Cluster 1:', view_cluster(0, train_image_directory, folder_name))
    print('Cluster 2:', view_cluster(1, train_image_directory, folder_name))
    print('Cluster 3:', view_cluster(2, train_image_directory, folder_name))
    print('Cluster 4:', view_cluster(3, train_image_directory, folder_name))
    print('Cluster 5:', view_cluster(4, train_image_directory, folder_name))

    return kmeans, kmeans.cluster_centers_, groups


def compare_with_other_trailer(model, testing_directory, centroids, trailer_clustered, trailer_testing, model_file,
                               model_weights):
    matrix = []

    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights)
    print("Loaded model from disk")

    # creates a ScandirIterator aliased as files
    with os.scandir(testing_directory) as files:
        # loops through each file in the directory
        for file in files:
            if file.name.endswith('.jpg'):
                # adds only the image files to the flowers list
                matrix.append(file.name)

    image_master_list = []

    data = {}
    # p = os.path.join(os.getcwd(), "interstellar_features_v2.pkl")
    # print(p)
    for images in matrix:

        image_master_dict = {'trailer_being_clustered': trailer_clustered,
                             'trailer_being_tested': trailer_testing,
                             'image_name': '',
                             'c0_distance': 0,
                             'c1_distance': 0,
                             'c2_distance': 0,
                             'c3_distance': 0,
                             'c4_distance': 0,
                             'shortest_distance': 0,
                             'closest_cluster': 0}
        # try to extract the features and update the dictionary
        feat = extract_features(images, loaded_model, testing_directory)
        # print(images)
        data[images] = feat
        image_master_dict['image_name'] = images
        # x = pca.transform(feat[0].reshape(1, -1))
        # kmeans.predict(x)

        for index, i in enumerate(centroids):
            if index == 0:
                image_master_dict['c0_distance'] = np.linalg.norm(feat - i)
            elif index == 1:
                image_master_dict['c1_distance'] = np.linalg.norm(feat - i)
            elif index == 2:
                image_master_dict['c2_distance'] = np.linalg.norm(feat - i)
            elif index == 3:
                image_master_dict['c3_distance'] = np.linalg.norm(feat - i)
            elif index == 4:
                image_master_dict['c4_distance'] = np.linalg.norm(feat - i)
            image_master_dict['shortest_distance'] = min(image_master_dict['c0_distance'],
                                                         image_master_dict['c1_distance'],
                                                         image_master_dict['c2_distance'],
                                                         image_master_dict['c3_distance'],
                                                         image_master_dict['c4_distance'])
            image_master_dict['closest_cluster'] = model.predict(feat)
        print(image_master_dict)
        image_master_list.append(image_master_dict)
    return image_master_list


def calculate_euclidean_distance(v1, v2):
    import math
    return math.dist(v1, v2)
