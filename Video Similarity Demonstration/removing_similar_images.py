
from scipy import spatial
import os
import cv2
from tensorflow.keras.models import Sequential, model_from_json



def get_vgg_similarity(image_1, image_2, basemodel):
    def get_feature_vector(img):
        img1 = cv2.resize(img, (224, 224))
        feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3), verbose='0')
        return feature_vector

    def calculate_similarity(vector1, vector2):
        return 1 - spatial.distance.cosine(get_feature_vector(vector1), get_feature_vector(vector2))

    vgg_score = calculate_similarity(image_1, image_2)
    return vgg_score


def remove_similar_images(name_list, model_file, model_weights):
    # load json and create model
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights)
    print("Loaded model from disk")

    length_of_total = len(name_list)
    i = 0
    while (i < length_of_total):
        print('Current Image: ', i)
        print('Remaining images:', length_of_total-i)
        deleted_names = []
        j = i + 1

        while j < length_of_total:

            image1 = cv2.imread(name_list[i])
            image2 = cv2.imread(name_list[j])
            vgg = get_vgg_similarity(image1, image2, loaded_model)
            if vgg > 0.8:
                print('Current Comparison: ', name_list[i], ' with ', name_list[j])
                print('structural similarity with ', j, 'th image= ', vgg)
                deleted_names.append(name_list[j])
            # print('J=',j)
            j += 1
        print('Deleted Image Names: ', deleted_names)
        for k in deleted_names:
            os.remove(k)
        # length_of_total = length_of_total-len(deleted_names)
        name_list = [x for x in name_list if x not in deleted_names]
        length_of_total = len(name_list)
        print('Updated Length is ', length_of_total, '\n')
        i += 1
