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
    momentsArray = np.empty([64,3]).astype(float)

    # Add moments for tiles row by row
    tileNumber = 0
    for tile in splitImage8x8:
        
        # Calculate mean (average).
        momentsArray[tileNumber, 0] = np.mean(tile)

        # Calculate standard deviation (sqrt of average of squared differences from the mean) (measure of data spread)
        momentsArray[tileNumber, 1] = np.std(tile)

        # Calculate skewness (asymetry of data distribution) (Fisher-Pearson used) (Alternate Simple calc mean-mode/sdev or 3(mean-median)/sdev)
        # (cube root of average of cubed differences from the mean)
        momentsArray[tileNumber, 2] = skew(tile, axis=None)

        tileNumber += 1

    return momentsArray

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

    print("Please enter desired  dimensionality reduction techniques\n(PCA, SVD, LDA, k-means): ")
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

def get_HOG_array(Xtype):

    directoryList = get_X_type_dir_list(Xtype)

    featuresList = []

    for path in directoryList:

        image = imread(path, as_gray=True)
        imageHOG, _ = calculateHOG(image)

        featuresList.append(imageHOG)
    return np.array(featuresList)


def get_ELBP_array(Xtype):

    directoryList = get_X_type_dir_list(Xtype)

    featuresList = []

    for path in directoryList:

        image = imread(path, as_gray=True)
        imageELBP, _ = calculateELBP(image)

        featuresList.append(imageELBP)
    return np.array(featuresList)

def build_C_matrix(modelName, XType):

    # featureArray will be nparray containing features row-wise.
    featureArray = None

    directoryList = get_X_type_dir_list(XType)

    if(modelName == 'CM'):
        # Color Moments
        pass
    if(modelName == 'ELBP'):
        # ELBP
        featureArray = get_ELBP_array(XType)
    if(modelName == 'HOG'):
        # HOG
        featureArray = get_HOG_array(XType)

    # Generate covariance-variance matrix
    cMatrix = np.covMatrix = np.cov(featureArray,bias=False)

    return cMatrix