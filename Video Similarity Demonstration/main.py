import time
from collections import OrderedDict

# for everything else
import pandas as pd

from extracting_images_from_trailer import *
from extraction_and_clustering import *
from removing_similar_images import *


def main():
    reference_movie = 'cast_away'
    recommended_movies = [
        # 'lord_of_the_flies',
        # 'nims_island',
        # 'six_days_seven_nights',
        # 'the_blue_lagoon',
        'the_most_dangerous_game',
        # 'the_red_turtle'
    ]

    all_movies = recommended_movies.copy()
    #all_movies.append(reference_movie)

    ## load all keyframes and process it
    for movie in all_movies:
        trailer = os.path.join(os.getcwd(), 'videos', movie + '_trailer.mp4')

        frame_names, image_directory = extract_frames(
            trailer,
            movie + '_key_frames', movie)

        remove_similar_images(frame_names, 'model_vgg.json', 'model_vgg.h5')

    ## Extract the centroid of all the clusters of the reference movie
    df_list = []
    distance = []

    kmeans, centroids, groups = get_folder_and_extract_features(
        os.path.join(os.getcwd(), reference_movie + '_key_frames'), reference_movie + '_key_frames', 'model_vgg.json',
        'model_vgg.h5')

    reference_movie_volumes = {}

    ## Percentage distribution of frames across clusters in the reference movie
    for i in groups:
        reference_movie_volumes[i] = len(groups[i]) / sum([len(j) for j in groups.values()])

    reference_movie_volumes_upadted = OrderedDict(sorted(reference_movie_volumes.items()))

    ## Calculating percentage distribution of each recommended movie
    for i in recommended_movies:
        directory = os.path.join(os.getcwd(), i + '_key_frames')
        image_data = compare_with_other_trailer(kmeans, directory, centroids, 'cast_away', i, 'model_vgg.json',
                                                'model_vgg.h5')
        df_list.append(pd.DataFrame(image_data))
        total_images_of_recommended_movie = 0
        for path in os.scandir(directory):
            if path.is_file():
                total_images_of_recommended_movie += 1
        temp_df = pd.DataFrame(image_data)
        temp_df['cluster_name'] = temp_df.apply(lambda x: x.closest_cluster[0], axis=1)
        temp_vector = temp_df.groupby('cluster_name')['image_name'].count().reset_index()
        temp_vector['perc'] = temp_vector.apply(lambda x: x.image_name / total_images_of_recommended_movie, axis=1)
        temp_vector['cluster_name'] = temp_vector['cluster_name'].astype('int64')
        while temp_vector.cluster_name.nunique() != 5:
            if 0 not in temp_vector.cluster_name.unique():
                temp_vector = temp_vector.append({'cluster_name': 0, 'image_name': 0, 'perc': 0}, ignore_index=True)
            elif 1 not in temp_vector.cluster_name.unique():
                temp_vector = temp_vector.append({'cluster_name': 1, 'image_name': 0, 'perc': 0}, ignore_index=True)
            elif 2 not in temp_vector.cluster_name.unique():
                temp_vector = temp_vector.append({'cluster_name': 2, 'image_name': 0, 'perc': 0}, ignore_index=True)
            elif 3 not in temp_vector.cluster_name.unique():
                temp_vector = temp_vector.append({'cluster_name': 3, 'image_name': 0, 'perc': 0}, ignore_index=True)
            else:
                temp_vector = temp_vector.append({'cluster_name': 4, 'image_name': 0, 'perc': 0}, ignore_index=True)
        temp_vector = temp_vector.sort_values(by='cluster_name')
        distance.append(calculate_euclidean_distance([i for i in reference_movie_volumes_upadted.values()],
                                                     temp_vector['perc'].to_list()))

    #Calculation of Video similarity score
    score = [1 / (i + 1) for i in distance]
    video_similarity_score = {}
    for index, i in enumerate(recommended_movies):
        video_similarity_score[i] = score[index]

    print(sorted([(value, key) for (key, value) in video_similarity_score.items()]))


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
