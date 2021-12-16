from featureUtilities import get_n_Most_Similar_Images

def task5():
	folder = input("Enter the folder for the query image\n")
	file = input("Enter a file name of the query image \n")

	latentsemantic = input("Enter latent semantics file \n")
	number = int(input("Enter number of images to return\n"))
	get_n_Most_Similar_Images(folder, file,latentsemantic, number)


