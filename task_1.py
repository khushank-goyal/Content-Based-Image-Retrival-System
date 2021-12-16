from featureUtilities import get_Subject_or_Type_Weight_Pairs

# Get user inputs.


def task1():
	model= (input("Enter the feature models: \n CM- Color Moments \n HOG- Histogram of Gradients\n ELBP- Extended Local Binary Pattern\n" ))

	subject= (input('Enter the type name :'))

	k = int(input("\nEnter the number of latent semantics 'k' you want :"))

	d = (input("\nEnter one of the dimensional reduction techniques below : \n PCA- Principal Component Analysis\n SVD- Singular Vector Decomposition\n"))

	get_Subject_or_Type_Weight_Pairs(model, subject, k, d, 0)


