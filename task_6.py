from featureUtilities import get_type_or_subject_of_image

def task6():
	folder = input("Enter the folder for the query image\n")
	file = input("Enter a file name of the query image \n")

	latentsemantic = input("Enter latent semantics file \n")


	get_type_or_subject_of_image(folder,file, latentsemantic, 0)

