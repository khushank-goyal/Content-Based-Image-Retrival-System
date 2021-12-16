

from featureUtilities import PCA, SVD, calculateColorMoments, calculateELBP, calculateHOG, euclidean, get_Features_based_on_Type_or_Subject, project_data, writeToFile, write_weight_pairs
import os
from skimage.io import imread
import numpy as np


def type_type_similarity(feature_model, k, dimentionality_reduction, directory):
    """
    This method creates the type-type similarity matrix using the given feature model, and also the latent
    semantics using the dimentionality reduction method provided.
    
    feature_model = CM, HOG, ELBP
    k = k latent semantics
    dimensional_reduction = PCA or SVD
    """

    # creating a dictionary to store images of each corresponding types
    type_dict = {
        'cc' : [],
        'rot': [],
        'neg': [],
        'poster': [],
        'noise01': [],
        'original': [],
        'emboss': [],
        'smooth': [],
        'noise02': [],
        'stipple': [],
        'con': [],
        'jitter': []
    }

    # for each image, append the feature_model matrix to the corresponding type in the type_dict
    for filename in os.listdir(directory):

        file = imread(directory + "/" + filename)

        # split the filename to get different attributes of the images
        a = filename.split('-')

        if feature_model == 'CM':
            type_dict[a[1]].append((a[2]+a[3], np.array(calculateColorMoments(file)).flatten()))
        elif feature_model == 'ELBP':
            type_dict[a[1]].append((a[2]+a[3], np.array(calculateELBP(file)[1]).flatten()))
        elif feature_model == 'HOG':
            type_dict[a[1]].append((a[2]+a[3], np.array(calculateHOG(file)[0]).flatten()))


    types = ['cc', 'rot', 'neg', 'poster', 'noise01', 'original', 'emboss', 'smooth', 'noise02', 'stipple', 'con', 'jitter']

    # create a type X type similarity matrix and initialize it with zeros
    similarity_matrix = [[0] * 12 for i in range(12)]

    # populate the similarity matrix by calculating the average distance of the images of the same subject, with different types
    for i in range(len(types)):
        for j in range(i+1,len(types)):
            total_distance = 0
            count = 0
            # for each image calculate distance
            for x in range(398):

                total_distance += euclidean(type_dict[types[i]][x][1], type_dict[types[j]][x][1])
                count += 1
            
            similarity_matrix[i][j] = total_distance/count
            similarity_matrix[j][i] = total_distance/count


    # save the similarity matrix to the sim_matrix directory
    similarity_matrix_folder = 'sim_matrix/'
    fileName = f'(type_type)_{feature_model}.txt'

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
    fileName = f'latent_semantics_(type_type)_{dimentionality_reduction}_{feature_model}_{k}.txt'

    filepath = f'{latent_semantics_folder}{fileName}'
    print(f'latent semantics stored at: {fileName}')
    write_weight_pairs(p_data,filepath,0)
    #writeToFile(p_data,filepath)

def task3():

    a= (input("Enter the feature models: \n CM- Color Moments \n ELBP- Local Binary patterns\n HOG- Histogram of Gradients\n" ))

    k = int(input("Enter the number of latent semantics 'k' you want :"))

    d = (input("\nEnter one of the dimensional reduction techniques below : \n PCA- Principal Component Analysis\n SVD- Singular Vector Decomposition\n"))

    directory = (input("\nEnter the path of the directory from which images are to be taken\n"))

    # directory = "/Users/khushank/Downloads/all/"


    type_type_similarity(a, k, d, directory)

# type_type_similarity(1, 8, 2)