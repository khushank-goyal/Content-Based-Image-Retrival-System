from featureUtilities import calculateELBP, calculateColorMoments, calculateHOG, euclidean, PCA, SVD, writeToFile, project_data, write_weight_pairs
import os
from skimage.io import imread
import numpy as np
import re


def feature_feature_similarity(feature_model, k, dimentionality_reduction, directory):

    type_dict = {
        'cc': 0,
        'rot': 1,
        'neg': 2,
        'poster': 3,
        'noise01': 4,
        'original': 5,
        'emboss': 6,
        'smooth': 7,
        'noise02': 8,
        'stipple': 9,
        'con': 10,
        'jitter': 11
    }
    # create a 3d matrix to store the images of each subject and initialize with 0s
    image_matrix = [[[0 for k in range(10)] for j in range(12)] for i in range(40)]

    for filename in os.listdir(directory):

        file = imread(directory +"/" +  filename)

        a = re.split('-|\.', filename)
        if feature_model == 'CM':
            image_matrix[int(a[2])-1][type_dict[a[1]]][int(a[3])-1] = np.array(calculateColorMoments(file)).flatten()
        elif feature_model == 'HOG':
            image_matrix[int(a[2])-1][type_dict[a[1]]][int(a[3])-1] = np.array(calculateHOG(file)[0]).flatten()
        elif feature_model == 'ELBP':
            image_matrix[int(a[2])-1][type_dict[a[1]]][int(a[3])-1] = np.array(calculateELBP(file)[1]).flatten()

    # initialize a 40 X 40 similarity matrix with 0s
    similarity_matrix = [[0] * 40 for i in range(40)]
    l = ['cc', 'rot', 'neg', 'poster', 'noise01', 'original', 'emboss', 'smooth', 'noise02', 'stipple', 'con', 'jitter']

    # populate the similarity matrix by calculating the average distance of the images of the same subject, with different types
    for i in range(40):
        for j in range(i+1, 40):
            summation = 0
            count = 0
            for x in range(12):
                for y in range(10):
                    summation += euclidean(image_matrix[i][x][y], image_matrix[j][x][y])
                    count += 1
            similarity_matrix[i][j] = summation/count
            similarity_matrix[j][i] = summation/count

    # save the similarity matrix to the sim_matrix directory
    similarity_matrix_folder = 'sim_matrix/'
    fileName = f'(subject_subject)_{feature_model}.txt'

    filepath = f'{similarity_matrix_folder}{fileName}'
    writeToFile(similarity_matrix,filepath)

    # perform dimensionality reduction on the similarity matrix using the provided dimensionality reduction method
    if dimentionality_reduction == 'PCA':
        _, right_feature_matrix = PCA(similarity_matrix, k)
    elif dimentionality_reduction == 'SVD':
        _, right_feature_matrix = SVD(similarity_matrix, k)

    # project the similarity matrix to the new latent semantic space
    p_data = project_data(right_feature_matrix,similarity_matrix)

    # save the laten semantics to the latent_semantic_files directory
    latent_semantics_folder = 'latent_semantic_files/'
    fileName = f'latent_semantics_(subject_subject)_{dimentionality_reduction}_{feature_model}_{k}.txt'

    filepath = f'{latent_semantics_folder}{fileName}'
    print(f'latent semantics stored at: {fileName}')
    write_weight_pairs(p_data,filepath,1)
    #writeToFile(p_data,filepath)


def task4():
    a= (input("Enter the feature models: \n CM- Color Moments \n HOG- Histogram of Gradients\n ELBP- Local Binary patterns\n" ))

    k = int(input("Enter the number of latent semantics 'k' you want :"))

    d = (input("\nEnter one of the dimensional reduction techniques below : \n PCA- Principal Component Analysis\n SVD- Singular Vector Decomposition\n"))

    directory = (input("\nEnter the path of the directory from which images are to be taken\n"))

    # directory = "/Users/khushank/Downloads/all/"



    feature_feature_similarity(a, k, d, directory)

# feature_feature_similarity('CM', 8, 'PCA')
