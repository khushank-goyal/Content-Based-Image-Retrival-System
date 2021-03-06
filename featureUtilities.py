import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.spatial.distance import euclidean, cityblock
from skimage.io import imread
from skimage import feature, exposure
from sklearn import preprocessing
import sys
import os
import re
import cv2
import matplotlib.pyplot as plt

def getAndSplitImage(idString):
    # Read corresponding image.
    image = imread('images_png/{}.png'.format(idString), as_gray=True)

    # Split horizontally 8 sections.
    splitImageHorizontal = np.array_split(image,8,axis=0)
    splitImage8x8 = []

    # Split vertically 8 sections to make 8x8
    for array in splitImageHorizontal:
        temp = np.array_split(array,8,axis=1)
        
        splitImage8x8 += temp

    return image, splitImage8x8
def SplitImage(image):
    

    # Split horizontally 8 sections.
    splitImageHorizontal = np.array_split(image,8,axis=0)
    splitImage8x8 = []

    # Split vertically 8 sections to make 8x8
    for array in splitImageHorizontal:
        temp = np.array_split(array,8,axis=1)
        
        splitImage8x8 += temp

    return splitImage8x8
def getAndSplitImageFromFile(path):
    # Read corresponding image.
    image = imread(path, as_gray=True)

    # Split horizontally 8 sections.
    splitImageHorizontal = np.array_split(image,8,axis=0)
    splitImage8x8 = []

    # Split vertically 8 sections to make 8x8
    for array in splitImageHorizontal:
        temp = np.array_split(array,8,axis=1)
        
        splitImage8x8 += temp

    return image, splitImage8x8

def testSplit(splitImage8x8):
    # Plot to test Split
    figure = plt.figure()

    tileNumber = 1
    rows = 8
    columns = 8

    for array in splitImage8x8:
        figure.add_subplot(rows, columns, tileNumber)
        print(array.ndim, array.shape)
        plt.imshow(array, cmap = 'gray')
        plt.title("tile_{}".format(tileNumber))
        plt.axis("off")
        tileNumber+=1

    plt.show()
    plt.clf()

def calculateColorMoments(splitImage8x8):
    # Create empty array of size 64x3
    #momentsArray = np.empty([64,3]).astype(float)
    means = []
    stds = []
    skews = []

    # Add moments for tiles row by row
    tileNumber = 0
    for tile in splitImage8x8:
        
        # Calculate mean (average).
        means.append(np.mean(tile))
        #momentsArray[tileNumber, 0] = np.mean(tile)

        # Calculate standard deviation (sqrt of average of squared differences from the mean) (measure of data spread)
        #momentsArray[tileNumber, 1] = np.std(tile)
        stds.append(np.std(tile))

        # Calculate skewness (asymetry of data distribution) (Fisher-Pearson used) (Alternate Simple calc mean-mode/sdev or 3(mean-median)/sdev)
        # (cube root of average of cubed differences from the mean)
        #momentsArray[tileNumber, 2] = skew(tile, axis=None)
        skews.append(skew(tile, axis=None))

        tileNumber += 1

    stds.extend(skews)
    means.extend(stds)

    return means

def momentsToHTML(momentsArray):
    dataFrame = pd.DataFrame(momentsArray, columns = ['Mean','SDEV','Skewness'])
    return dataFrame.to_html()

def calculateELBP(image):
    # Calculate rotational invarient (method uniform is greyscale rotational invarient) lbp with 8 points and radius 1
    points = 32
    radius = 4
    RILocalBinaryPatterns = feature.local_binary_pattern(image, points, radius, method="uniform")
    # print(RILocalBinaryPatterns)

    

    # number of unique values in lbp
    numBins = int(RILocalBinaryPatterns.max() + 1)
    # create histogram counting features to create feature vector, (density=true normalizes hist)
    (lbpHistogram, _) = np.histogram(RILocalBinaryPatterns, density=True,
                bins=numBins,
                range=(0, numBins))

    return lbpHistogram, RILocalBinaryPatterns

def calculateWindowedELBP(splitImage8x8):
    lbpHistogramWidowed = []
    RILocalBinaryPatterns = "output unused in windowed"

    for tile in splitImage8x8:
        # Calculate rotational invarient (method uniform is greyscale rotational invarient) lbp with 8 points and radius 1
        points = 8
        radius = 1
        RILocalBinaryPatterns = feature.local_binary_pattern(tile, points, radius, method="uniform")
        # print(RILocalBinaryPatterns)

        

        # number of unique values in lbp
        numBins = int(RILocalBinaryPatterns.max() + 1)
        # create histogram counting features to create feature vector, (density=true normalizes hist)
        (lbpHistogram, _) = np.histogram(RILocalBinaryPatterns, density=True,
                    bins=numBins,
                    range=(0, numBins))

        np.concatenate((lbpHistogramWidowed, lbpHistogram))

    return lbpHistogramWidowed, RILocalBinaryPatterns

def LBPImage(RILocalBinaryPatterns, id):
    # LBP image representation
    plt.figure('Texture')
    plt.imshow(RILocalBinaryPatterns, cmap = 'gray')
    plt.savefig('HTML_images/{}_LBPImage.png'.format(id))
    # plt.show()
    plt.clf()
    return '<img src="../HTML_images/{}_LBPImage.png">'.format(id)

def LBPHistogramImage(RILocalBinaryPatterns, id):
    # Plot histogram of lbp features
    plt.figure('Texture Hist')
    # number of unique values in lbp
    numBins = int(RILocalBinaryPatterns.max() + 1)
    plt.hist(RILocalBinaryPatterns.ravel(), density=True, bins=numBins, range=(0, numBins), facecolor='0.5')
    plt.savefig('HTML_images/{}_LBPHistogram.png'.format(id))
    # plt.show()
    plt.clf()
    return '<img src="../HTML_images/{}_LBPHistogram.png">'.format(id)

def calculateHOG(image):
    # multichannel set to false to indicate grayscale, L2-Hys indicated l2-norm with .2 max
    featureVector, hogVisualized = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), block_norm="L2-Hys", visualize=True, multichannel=False)
    
    return featureVector, hogVisualized

def HOGImage(hogVisualized, id):
    # Rescale and print visualization for verification.
    hogVizRescale = exposure.rescale_intensity(hogVisualized, in_range=(0, 10))

    plt.figure('Texture Hist')
    plt.imshow(hogVizRescale, cmap = 'gray')
    plt.savefig('HTML_images/{}_HOGImage.png'.format(id))
    # plt.show()
    plt.clf()
    return '<img src="../HTML_images/{}_HOGImage.png">'.format(id)

#######################################################################









def imageHTMLreport(folderName, path, iteration):
    # Print full array
    np.set_printoptions(threshold=sys.maxsize)

    ########################################################################
    ######################### Get and Split Image ##########################
    ########################################################################


    

    # Get and split image.
    image, splitImage8x8 = getAndSplitImageFromFile(path)

    # Test Split 
    # testSplit(splitImage8x8)




    ########################################################################
    ######################### Calculate Color Moments ######################
    ########################################################################

    # Calculate color moments array
    momentsArray = calculateColorMoments(splitImage8x8)
    # print(momentsArray)
    momentsHTML = momentsToHTML(momentsArray)




    ########################################################################
    ######################### Calculate ELBP ###############################
    ########################################################################

    # Calculate ELBP
    lbpHistogramArray, RILocalBinaryPatterns = calculateELBP(image)

    # print(lbpHistogramArray)

    # LBP image representation
    LBPImage_tag = LBPImage(RILocalBinaryPatterns, folderName + '_' + iteration)

    # Plot histogram of LBP features
    LBPHistogramImage_tag = LBPHistogramImage(RILocalBinaryPatterns, folderName + '_' + iteration)




    ########################################################################
    ################## Histograms of Oriented Gradients ####################
    ########################################################################

    # Calculate HOG
    HOGFeatureVector, hogVisualized = calculateHOG(image)

    # print(HOGFeatureVector)

    #  HOG image representation
    HOGImage_tag = HOGImage(hogVisualized, folderName + '_' + iteration)




    ########################################################################
    ################### Generate Human-Readable Report #####################
    ########################################################################

    #Generating HTML

    heading = '<h1 style="font-size:20px;"> Image Report for ' + path + ' </h1>'


    header = '<div class="top">' + heading +'</div> \n'

    ogImageTag = '<img src="../{}">'.format(path)
    content0 = '<div class="og"> '+ ogImageTag +' </div> \n'

    subheading1 = '<h2> Color Moments </h2> \n'
    content1 = '<div class="table"> ' + momentsHTML + ' </div> \n'


    subheading2 = '<h2> ELBP </h2> \n'
    content2 = ('<div class="array"> '+ repr(lbpHistogramArray) +' </div> \n'
                '<div class="representation"> '+ LBPImage_tag +' </div> \n'
                '<div class="histogram"> '+ LBPHistogramImage_tag +' </div> \n'
                )

    subheading3 = '<h2> HOG </h2> \n'
    content3 = ('<div class="array"> '+ repr(HOGFeatureVector) +' </div> \n'
                '<div class="representation"> '+ HOGImage_tag +' </div> \n'
                )

    html = header + content0 + subheading1 + content1 + subheading2 + content2 + subheading3 + content3
    
    return html

    #######################################################################



def searchSetupSingleModel():
    # Input prompts.
    print("Please enter folder name in working directory: ")
    folderName = input()

    print("Please enter image name in provided folder without extension: ")
    imageName = input()

    print("Please enter model name ('CM', 'ELBP', 'HOG') without quotations: ")
    modelName = input()

    print("Please enter number of matches desired: ")
    matchesNum = int(input())


    directoryList = []


    

    # Get base image.
    baseImagePath = '{}\{}.png'.format(folderName, imageName)
    baseImage = imread(baseImagePath, as_gray=True)

    # Add all file paths under directory to list. 
    for path in os.listdir(folderName):

        fullPath = os.path.join(folderName, path)
        if (fullPath != baseImagePath):
            directoryList.append(fullPath)

    return folderName, imageName, baseImage, baseImagePath, modelName, matchesNum, directoryList


def searchSetupMultiModel():
    # Input prompts.
    print("Please enter folder name in working directory: ")
    folderName = input()

    print("Please enter image name in provided folder without extension: ")
    imageName = input()

    print("Please enter number of matches desired: ")
    matchesNum = int(input())


    directoryList = []


    

    # Get base image.
    baseImagePath = '{}\{}.png'.format(folderName, imageName)
    baseImage = imread(baseImagePath, as_gray=True)

    # Add all file paths under directory to list. 
    for path in os.listdir(folderName):

        fullPath = os.path.join(folderName, path)
        if (fullPath != baseImagePath):
            directoryList.append(fullPath)

    return folderName, imageName, baseImage, baseImagePath, matchesNum, directoryList


def momentsRanking(baseImagePath, directoryList):

    similarityList = []
    rankingsList = []

    for compImagePath in directoryList:
        
        # Get moments
        _, baseImageSplit = getAndSplitImageFromFile(baseImagePath)
        
        baseImageMoments = calculateColorMoments(baseImageSplit)

        _, compImageSplit = getAndSplitImageFromFile(compImagePath)
        
        compImageMoments = calculateColorMoments(compImageSplit)

        # # Calculate difference.
        difference = np.absolute(np.subtract(baseImageMoments, compImageMoments))

        # # Calculate sum. Lower is better.
        similarity = sum(map(sum,difference)) 
        

        similarityList.append(similarity)


    # Normalize
    similarityList = preprocessing.minmax_scale(similarityList)
    
    # Add path and similarity to rankings
    for index in range(len(directoryList)):
        tuple = (directoryList[index], similarityList[index])
        rankingsList.append(tuple)

    return rankingsList

def klDivergence(histA, histB):
  # kullback leibler divergence
  a = np.asarray(histA)
  b = np.asarray(histB)
  
  # Where same.
  same = np.logical_and(a != 0, b != 0)
  
  return np.sum(a[same] * np.log2(a[same] / b[same]))


def ELBPRanking(baseImage, directoryList):

    similarityList = []
    rankingsList = []

    # Get ELBPS
    baseImageHOG, _ = calculateELBP(baseImage)
    
    for compImagePath in directoryList:
                
        compImage = imread(compImagePath, as_gray=True)
        compImageHOG, _ = calculateELBP(compImage)

        
        # Calc similarity with kullback leibler divergence (smaller is better)
        # similarity = klDivergence(baseImageHOG, compImageHOG)

        # Calc similarity with euclidean (smaller is better)
        similarity = euclidean(baseImageHOG, compImageHOG)
        
        print("image={},,,SIM={}".format(compImagePath, similarity))

        # Add to list.
        similarityList.append(similarity)


    # Normalize
    similarityList = preprocessing.minmax_scale(similarityList)
    
    # Add path and similarity to rankings
    for index in range(len(directoryList)):
        tuple = (directoryList[index], similarityList[index])
        rankingsList.append(tuple)

    return rankingsList

def ELBPRankingWindowed(baseImage, baseImagePath, directoryList):

    similarityList = []
    rankingsList = []

    # Get ELBPS
    _,baseImage = getAndSplitImageFromFile(baseImagePath)
    baseImageHOG, _ = calculateWindowedELBP(baseImage)
    
    for compImagePath in directoryList:
                
        _,compImage = getAndSplitImageFromFile(compImagePath)
        compImageHOG, _ = calculateWindowedELBP(compImage)

        
        # Calc similarity with kullback leibler divergence (smaller is better)
        # similarity = klDivergence(baseImageHOG, compImageHOG)

        # Calc similarity with euclidean (smaller is better)
        similarity = euclidean(baseImageHOG, compImageHOG)
        
        print("image={},,,SIM={}".format(compImagePath, similarity))

        # Add to list.
        similarityList.append(similarity)


    # Normalize
    similarityList = preprocessing.minmax_scale(similarityList)
    
    # Add path and similarity to rankings
    for index in range(len(directoryList)):
        tuple = (directoryList[index], similarityList[index])
        rankingsList.append(tuple)

    return rankingsList

def HOGRanking(baseImage, directoryList):

    similarityList = []
    rankingsList = []

    # Get HOGS
    baseImageHOG, _ = calculateHOG(baseImage)
    
    for compImagePath in directoryList:
                
        compImage = imread(compImagePath, as_gray=True)
        compImageHOG, _ = calculateHOG(compImage)

        
        # Calc similarity with kullback leibler divergence (smaller is better)
        # similarity = klDivergence(baseImageHOG, compImageHOG)

        # Calc similarity with manhattan distance (smaller is better)
        similarity = cityblock(baseImageHOG, compImageHOG)
        
        print("image={},,,SIM={}".format(compImagePath, similarity))

        # Add to list.
        similarityList.append(similarity)


    # Normalize
    similarityList = preprocessing.minmax_scale(similarityList)
    
    # Add path and similarity to rankings
    for index in range(len(directoryList)):
        tuple = (directoryList[index], similarityList[index])
        rankingsList.append(tuple)

    return rankingsList


################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
                                                                                                                                                                                 
                                                                                                                                                                                  
# PPPPPPPPPPPPPPPPP   hhhhhhh                                                                        TTTTTTTTTTTTTTTTTTTTTTT                                                        
# P::::::::::::::::P  h:::::h                                                                        T:::::::::::::::::::::T                                                        
# P::::::PPPPPP:::::P h:::::h                                                                        T:::::::::::::::::::::T                                                        
# PP:::::P     P:::::Ph:::::h                                                                        T:::::TT:::::::TT:::::T                                                        
#   P::::P     P:::::P h::::h hhhhh         aaaaaaaaaaaaa      ssssssssss       eeeeeeeeeeee         TTTTTT  T:::::T  TTTTTTwwwwwww           wwwww           wwwwwww ooooooooooo   
#   P::::P     P:::::P h::::hh:::::hhh      a::::::::::::a   ss::::::::::s    ee::::::::::::ee               T:::::T         w:::::w         w:::::w         w:::::woo:::::::::::oo 
#   P::::PPPPPP:::::P  h::::::::::::::hh    aaaaaaaaa:::::ass:::::::::::::s  e::::::eeeee:::::ee             T:::::T          w:::::w       w:::::::w       w:::::wo:::::::::::::::o
#   P:::::::::::::PP   h:::::::hhh::::::h            a::::as::::::ssss:::::se::::::e     e:::::e             T:::::T           w:::::w     w:::::::::w     w:::::w o:::::ooooo:::::o
#   P::::PPPPPPPPP     h::::::h   h::::::h    aaaaaaa:::::a s:::::s  ssssss e:::::::eeeee::::::e             T:::::T            w:::::w   w:::::w:::::w   w:::::w  o::::o     o::::o
#   P::::P             h:::::h     h:::::h  aa::::::::::::a   s::::::s      e:::::::::::::::::e              T:::::T             w:::::w w:::::w w:::::w w:::::w   o::::o     o::::o
#   P::::P             h:::::h     h:::::h a::::aaaa::::::a      s::::::s   e::::::eeeeeeeeeee               T:::::T              w:::::w:::::w   w:::::w:::::w    o::::o     o::::o
#   P::::P             h:::::h     h:::::ha::::a    a:::::assssss   s:::::s e:::::::e                        T:::::T               w:::::::::w     w:::::::::w     o::::o     o::::o
# PP::::::PP           h:::::h     h:::::ha::::a    a:::::as:::::ssss::::::se::::::::e                     TT:::::::TT              w:::::::w       w:::::::w      o:::::ooooo:::::o
# P::::::::P           h:::::h     h:::::ha:::::aaaa::::::as::::::::::::::s  e::::::::eeeeeeee             T:::::::::T               w:::::w         w:::::w       o:::::::::::::::o
# P::::::::P           h:::::h     h:::::h a::::::::::aa:::as:::::::::::ss    ee:::::::::::::e             T:::::::::T                w:::w           w:::w         oo:::::::::::oo 
# PPPPPPPPPP           hhhhhhh     hhhhhhh  aaaaaaaaaa  aaaa sssssssssss        eeeeeeeeeeeeee             TTTTTTTTTTT                 www             www            ooooooooooo   
                                                                                                                                                                                  

################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

def task_one_prompts():
    
    # Input prompts.

    print("Please enter model name ('CM', 'ELBP', 'HOG') without quotations: ")
    modelName = input()

    print("Please enter desired X image type\n(cc, con, detail, emboss, jitter, neg, noise1, noise2,\noriginal, poster, rot, smooth, stipple): ")
    XType = input()

    print("Please enter desired k value: ")
    kValue = int(input())

    print("Please enter desired  dimensionality reduction techniques\n(, SVD, LDA, k-means): ")
    drTechnique = input()

    return modelName, XType, kValue, drTechnique


def get_X_type_dir_list(XType, folderName = 'database'):

    directoryList = []

    # Add all matching file paths under directory to list. 
    for path in os.listdir(folderName):

        fullPath = os.path.join(folderName, path)
        if (XType in fullPath):
             directoryList.append(fullPath)
        

    return directoryList


def get_Y_type_dir_list(YType, folderName = 'database'):

    directoryList = []
    YType = "-"+str(YType)+"-"

    # Add all matching file paths under directory to list. 
    for path in os.listdir(folderName):

        fullPath = os.path.join(folderName, path)
        if (YType in fullPath):
             directoryList.append(fullPath)
        

    return directoryList

def writeToFile(vectors,file_name):
    np.savetxt(file_name,vectors,delimiter=',')

def readFromFile(file_name):
    with open(file_name) as file:
        newVectors = [[complex(digit) for digit in line.split(',')] for line in file]

    return newVectors

def get_New_Image(folderName,imageName):
   
    # Get new image.
    baseImagePath = f'{folderName}/{imageName}'
    baseImage = imread(baseImagePath, as_gray=True)

    return baseImage

def displayImages(images):
    '''
    images: 2D array of all of the images from the provided folder

    values: Indexes of the images that should be displayed

    Functionality: Displays the images vertically by similarity score with a higher
        similarity at the top. 
    '''
    if (len(images))==1:
        plt.imshow(images[0])

    else:
        _,imageDisplay = plt.subplots(len(images),1)

        for index in range(len(images)):
            imageDisplay[index].imshow(images[index],cmap='gray')

    plt.show()

def get_CM_array(directoryList):
    featuresList = []

    # Calculate CM for each image in list, add to featuresList.
    for path in directoryList:

        image = imread(path, as_gray=True)
        imageCM = calculateColorMoments(image)

        featuresList.append(imageCM)
    # Return as array.
    return np.array(featuresList)

def get_HOG_array(directoryList):
    featuresList = []

    # Calculate HOG for each image in list, add to featuresList.
    for path in directoryList:

        image = imread(path, as_gray=True)
        imageHOG, _ = calculateHOG(image)

        featuresList.append(imageHOG)
    # Return as array.
    return np.array(featuresList)


def get_ELBP_array(directoryList):

    featuresList = []
    # Calculate ELBP for each image in list, add to featuresList, return as array.
    for path in directoryList:

        image = imread(path, as_gray=True)
        imageELBP, _ = calculateELBP(image)

        featuresList.append(imageELBP)
    # Return as array.
    return np.array(featuresList)


# X_or_Y = 0 for X and  X_or_Y = 1 for Y
def get_feature_array(feature_model_name,type_name,X_or_Y):
    #directoryList = []
    if X_or_Y == 0:
        directoryList = get_X_type_dir_list(type_name)
    elif X_or_Y == 1:
        directoryList = get_Y_type_dir_list(type_name)

    if(feature_model_name == 'CM'):
        # Color Moments
        featureArray = get_CM_array(directoryList)
    if(feature_model_name == 'ELBP'):
        # ELBP
        featureArray = get_ELBP_array(directoryList)
    if(feature_model_name == 'HOG'):
        # HOG
        featureArray = get_HOG_array(directoryList)

    return featureArray



def build_C_matrix(featureArray):

    cMatrix = np.cov(np.array(featureArray).T)

    return cMatrix   


def eigen_decomp(matrix):

    # Perform eigendecomposition. 
    eigValues, eigVectors = np.linalg.eig(matrix)

    return eigValues, eigVectors

def extract_K_values(eigValues, eigVectors, kValue):
    valueIndexList = []
    

    for index,value in enumerate(eigValues):
        valueIndexList.append((index,value))
    
    # Sort to find contribution.
    valueIndexList.sort(key = lambda x: x[1], reverse=True)





    k_highest_vectors = []

    for index,_ in valueIndexList:
        if index < kValue:
            k_highest_vectors.append(eigVectors[index])

    #k_highest_vectors = df_eigVectors.loc[:, highestIndexList]#[0:kValue]
    k_highest_values = [value for index,value in valueIndexList][0:kValue]


    return k_highest_values, k_highest_vectors
    


def generate_data_for_PCA(modelName, type_name, X_or_Y):

    featureArray = get_feature_array(modelName,type_name,X_or_Y)

    return featureArray

def generate_data_and_perform_PCA(modelName, type_name, X_or_Y, kValue):
    featureArray = generate_data_for_PCA(modelName, type_name, X_or_Y)
    return PCA(featureArray,kValue)


def PCA(featureArray, kValue):
    '''
    featureArray: original data matrix
    kValue: k latent semantics
    orig_x_dimension: just put len of a row for a similarity matrix, otherwise it would be dertemined when generating the feature data
    orig_y_dimension: just put len of the similarity matrix, otherwise it would be dertemined when generating the feature data

    '''

    # Generate covariance matrix.
    cMatrix = build_C_matrix(featureArray)
    

    # Perform eigendecomposition. 
    eigValues, eigVectors = eigen_decomp(cMatrix)

   

    inverse_eigen = np.array(eigVectors).T



    #print(len(eigVectors),len(eigVectors[0]))
    #Need to figure out how to cut down the rows and columns to be equal number

    if kValue > len(cMatrix):
    	kValue = len(cMatrix)


    # Pick top k values and vectors. 
    #k_highest_values, k_highest_vectors = extract_K_values(eigValues, eigVectors, kValue)

    k_highest_values, k_highest_inverse_vectors = extract_K_values(eigValues, inverse_eigen, kValue)


    return k_highest_values, k_highest_inverse_vectors


def generate_data_and_perform_SVD(modelName, XType, X_or_Y, kValue):
    featureArray = get_feature_array(modelName, XType, X_or_Y)
    return SVD(featureArray,kValue)


def SVD(featureArray, kValue):


    if len(featureArray[0])<kValue:
        kValue = len(featureArray[0])
    
    u, s, v = SVDHelper(featureArray)
    #u, s, v = np.linalg.svd(featureArray)
    

    k_highest_values, k_highest_vectors = extract_K_values(s,u,kValue)

    _, k_highest_right_vectors = extract_K_values(s,v,kValue)
   

    return k_highest_values, k_highest_right_vectors


def SVDHelper(a):
    aTranspose1 = a @ np.transpose(a) 
    S, U = eigen_decomp(aTranspose1)
    aTranspose2 = np.transpose(a) @ a
    eigenValues, V = eigen_decomp(aTranspose2)
    V = np.transpose(V)
    S = np.sqrt(S)

    #returns U, S, V where S is the eigenValues and U and V are eigenvectors 
    return U, S, V
    



################################
#For task 1 or 2

################################################################################################ 
def get_Features_based_on_Type_or_Subject(feature_model_name,X_or_Y):
	types = ['cc', 'rot', 'neg', 'poster', 'noise01', 'original', 'emboss', 'smooth', 'noise02', 'stipple', 'con', 'jitter']
	subjects = [x for x in range(1,41)]
	grouping_vectors = []

	if X_or_Y == 0:
		for t in types:
			feature_array = get_feature_array(feature_model_name,t,X_or_Y)


			#Using Average for Now

			type_vector = []
			for j in range(len(feature_array[0])):
				avg = 0
				for row in feature_array:
					avg+=row[j]

				avg = avg/(len(feature_array))
				type_vector.append(avg)

			grouping_vectors.append(type_vector)

	elif X_or_Y == 1:
		for t in subjects:
			#print(t)
			feature_array = get_feature_array(feature_model_name,t,X_or_Y)
			#print(len(feature_array))
			#print(len(featureArray[0]))

			#Using Average for Now

			type_vector = []
			for j in range(len(feature_array[0])):
				avg = 0
				for row in feature_array:
					avg+=row[j]

				avg = avg/(len(feature_array))
				type_vector.append(avg)

			grouping_vectors.append(type_vector)
			

	return grouping_vectors

def write_weight_pairs(new_projected_data, fileName, x_or_y):
    types = ['cc', 'rot', 'neg', 'poster', 'noise01', 'original', 'emboss', 'smooth', 'noise02', 'stipple', 'con', 'jitter']
    subjects = [x for x in range(1,41)]
    f = open(fileName, 'w')
  # python will convert \n to os.linesep

    for i in range(len(new_projected_data[0])):
        values = [(row[i],index+1) for index,row in enumerate(new_projected_data)]
        #print(values)

        values.sort(key = lambda x: x[0], reverse=True)

        f.write(f'\n\nlatent_semantics {i+1}\n')

        if x_or_y == 0:
            for value in values:
                f.write(f'{types[value[1]]}, {value[0]}\n')
        else:
            for value in values:
                f.write(f'{value[1]}, {value[0]}\n')

    f.close()




def project_data(right_feature_matrix, grouping_vectors):
	new_projected_data = np.matmul(grouping_vectors, np.array(right_feature_matrix).T)
	return new_projected_data

#Task 1 and 2 call this
def get_Subject_or_Type_Weight_Pairs(feature_model, type_name, k, dimensional_reduction_technique, X_or_Y=0):
    '''
    feature_model = CM, HOG, ELBP
    type_name = X (cc,con...) or Y (1,2..)
    k = k latent semantics
    dimensional_reduction_technique = PCA or SVD
    X_or_Y = 0 if is an X value or 1 if it is a Y value
    '''

    # all latent semantic files can be stored in the latent_semantics_folder
    latent_semantics_folder = 'latent_semantic_files/'

    eigen_vectors = []
    right_feature_matrix = []

    if dimensional_reduction_technique == 'PCA':
        _,right_feature_matrix = generate_data_and_perform_PCA(feature_model,type_name,X_or_Y,k)
    elif dimensional_reduction_technique == 'SVD':
        _,right_feature_matrix = generate_data_and_perform_SVD(feature_model,type_name,X_or_Y,k)


    ####################################################################################
    opposite = 0
    if (X_or_Y == 0):
        opposite = 1


    grouping_vectors = get_Features_based_on_Type_or_Subject(feature_model, opposite)

    p_data = project_data(right_feature_matrix,grouping_vectors)

    ####################################################################################
    #print(len(p_data),len(p_data[0]))

    fileName = f'latent_semantics_{dimensional_reduction_technique}_{feature_model}_{type_name}_{k}.txt'

    fileName2 = f'rlatent_semantics_{dimensional_reduction_technique}_{feature_model}_{type_name}_{k}.txt'

    filepath = f'{latent_semantics_folder}{fileName}'

    filepath2 = f'{latent_semantics_folder}{fileName2}'

    write_weight_pairs(p_data,filepath,opposite)
    #writeToFile(p_data,filepath)
    writeToFile(right_feature_matrix,filepath2)

    print(f'latent semantics stored at: {filepath}')


######################################
#For task 5
# Definiton: Loads all the images from a folder
# folder = folder to get images from
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        b, g, r = cv2.split(img)
        if img is not None:
            images.append(b)
    return images

# Definiton: Loads all the image names from a folder
# folder = folder to get image names from
def load_image_names_from_folder(folder):
    imageNames = []
    for filename in os.listdir(folder):
        imageNames.append(filename)
    return imageNames


# Definiton: Returns the feature for a given image based on the featureType given
# featureType = HOG, CM or ELBP
# iamge = image to get features from
def getImageFeature(featureType, image):

    if featureType == "HOG":
        new_image, _ = calculateHOG(image)
    elif featureType == "CM":
        new_image = calculateColorMoments(SplitImage(image))
    elif featureType == "ELBP":
        new_image, _= calculateELBP(image)
    return new_image

# Runs through all images in database and returns the similarity score between the given image similarity and all the images in the database
# Inputs:
# allImages = All the images in the database
# baseSimilarity = similarity score for given image in task 5
# featureType = feature type
# semantic_data = semantic data from given file

def get_Image_Latent_Semantic_Similarity(allImages,baseSimilarity,featureType, semantic_data):
    dataBaseSimilarities = []
    index = 0
    for image in allImages:
        imageFeature = getImageFeature(featureType, image)
        imageSimilarity = np.dot([imageFeature],np.transpose(semantic_data) )
        
        dataBaseSimilarities.append( (index,np.linalg.norm(baseSimilarity-imageSimilarity)))
        index += 1
    return dataBaseSimilarities


#
# folderName = name of folder that given image is located
# iamgeName = name of image file for given query image
# fileName = name of latent semantic file located in latent_semantic_files folder
# n = number of most similar images to return
#

#Task 5
def get_n_Most_Similar_Images(folderName,imageName, fileName, n):
    #get image given image
    new_image = get_New_Image(folderName,imageName)

 
    latent_semantics_folder = 'latent_semantic_files/'
    filepath = f'{latent_semantics_folder}r{fileName}'

    #get given latent semantic file name
    semantic_data = readFromFile(filepath)
    allImageNames = load_image_names_from_folder("database")
    allImages = load_images_from_folder("database")

    #split latent semantic file based on name to get information about file
    latentSemanticsName = fileName.split("_")
    dimesionalityReductionTechnique = latentSemanticsName[2]
    featureType = latentSemanticsName[3]

    #get the features of given image based on the latent semantic file feature type
    new_image = getImageFeature(featureType, new_image)

    
    baseSimilarity = np.dot([new_image],np.transpose(semantic_data))

    dataBaseSimilarities = get_Image_Latent_Semantic_Similarity(allImages, baseSimilarity,featureType, semantic_data)

    
    dataBaseSimilarities.sort(key = lambda x: x[1], reverse=False)
    outputImages = []
    outputNames = []
    for inc in range(n):
        outputImages.append(allImages[dataBaseSimilarities[inc][0]])
        outputNames.append(allImageNames[dataBaseSimilarities[inc][0]])
    print(outputNames)
    displayImages(outputImages)

#def get_Most_Similar_Image_Type()


######################################
#For task 6

# Definition: Runs through types in featureType and returns the similarity score between the given image similarity that type
# Inputs:
# typeFeatures = calculated features for each type in database
# baseSimilarity = similarity score for given image in task 6 
# featureType = feature type
# semantic_data = semantic data from given file
def get_Type_Latent_Semantic_Similarity(typeFeatures,baseSimilarity,featureType, semantic_data):
    dataBaseSimilarities = []
    index = 0
    for type in typeFeatures:
        
        imageSimilarity = np.dot([type],np.transpose(semantic_data) )
        
        dataBaseSimilarities.append( (index,np.linalg.norm(baseSimilarity-imageSimilarity)))
        index += 1
    return dataBaseSimilarities


#
# folderName = name of folder that given image is located
# iamgeName = name of image file for given query image
# fileName = name of latent semantic file located in latent_semantic_files folder
# x_or_y = whether it is type of image for task 6(0) or subject of image for task 7(1)
# 
def get_type_or_subject_of_image(folderName,imageName, fileName, x_or_y):
    types = ['cc', 'rot', 'neg', 'poster', 'noise01', 'original', 'emboss', 'smooth', 'noise02', 'stipple', 'con', 'jitter']
    subjects = [x for x in range(1,41)]
    #get image given image
    new_image = get_New_Image(folderName,imageName)

 
    latent_semantics_folder = 'latent_semantic_files/'
    filepath = f'{latent_semantics_folder}r{fileName}'

    #get given latent semantic file name
    semantic_data = readFromFile(filepath)

    #split latent semantic file based on name to get information about file
    latentSemanticsName = fileName.split("_")
    dimesionalityReductionTechnique = latentSemanticsName[2]
    featureType = latentSemanticsName[3]

    #get the features of given image based on the latent semantic file
    new_image = getImageFeature(featureType, new_image)
    if x_or_y == 0:
        typeFeatures = get_Features_based_on_Type_or_Subject(featureType, 0)

    elif x_or_y == 1:
        typeFeatures = get_Features_based_on_Type_or_Subject(featureType, 1)



    baseSimilarity = np.dot([new_image],np.transpose(semantic_data))

    dataBaseSimilarities = get_Type_Latent_Semantic_Similarity(typeFeatures, baseSimilarity,featureType, semantic_data)

    # sort similarites 
    dataBaseSimilarities.sort(key = lambda x: x[1], reverse=False)

    outputNames = []
    for inc in range(len(dataBaseSimilarities)):
        if x_or_y == 0:
            outputNames.append(types[dataBaseSimilarities[inc][0]])
        elif x_or_y == 1:
            outputNames.append(subjects[dataBaseSimilarities[inc][0]])

    print(outputNames)




