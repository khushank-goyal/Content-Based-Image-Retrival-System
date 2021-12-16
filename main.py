import sys
from featureUtilities import get_Subject_or_Type_Weight_Pairs, get_n_Most_Similar_Images, get_type_or_subject_of_image
from task_1 import task1
from task_2 import task2
from task_3 import task3
from task_4 import task4
from task_5 import task5
from task_6 import task6
from task_7 import task7
from task_8 import task8


def main():


	args = sys.argv

	if len(args) == 2:
		task = args[1]
		if task == 't1':
			task1()
		elif task == 't2':
			task2()
		elif task == 't3':
			task3()
		elif task == 't4':
			task4()
		elif task == 't5':
			task5()
		elif task == 't6':
			task6()
		elif task == 't7':
			task7()
		elif task == 't8':
			task8()
		else:
			help_message()
	else:
		help_message()


def help_message():
	message = '''
	To perform task 1:
	\tpython ./main.py t1

	\tFollowed by providing:
	\t\t feature model (CM, ELBP, HOG)
	\t\t specified X value (cc, con, detail, emboss, jitter, neg, noise1, noise2, original, poster, rot, smooth, stipple)
	\t\t specified k value (1-192 for CM, 1-32 for ELBP, 1-1700 for HOG)
	\t\t specified dimensionality reduction technique (SVD, PCA)

	Data will be written to latent_semantics_(dimensional_reduction_technique)_(feature_model)_(type_name)_(k).txt

	To perform task 2:
	\tpython ./main.py t2

	\tFollowed by providing:
	\t\t feature model (CM, ELBP, HOG)
	\t\t specified Y value (1-40)
	\t\t specified k value (1-192 for CM, 1-32 for ELBP, 1-1700 for HOG)
	\t\t specified dimensionality reduction technique (SVD, PCA)

	Data will be written to latent_semantics_(dimensional_reduction_technique)_(feature_model)_(subject_number)_(k).txt

	To perform task 3:
	\tpython ./main.py t3

	\tFollowed by providing:
	\t\t feature model (CM, ELBP, HOG)
	\t\t specified k value (1-12)
	\t\t specified dimensionality reduction technique (SVD, PCA)
	\t\t folder name (database)

	To perform task 4:
	\tpython ./main.py t4

	\tFollowed by providing:
	\t\t feature model (CM, ELBP, HOG)
	\t\t specified k value (1-40)
	\t\t specified dimensionality reduction technique (SVD, PCA) 
	\t\t folder name (database)

	To perform task 5:
	\tpython ./main.py t5

	\tFollowed by providing:
	\t\t local folder name ('database' for an image already provided, 'images' for a new image )
	\t\t image name
	\t\t semantic file name
	\t\t specified n value 

	To perform task 6:
	\tpython ./main.py t6

	\tFollowed by providing:
	\t\t local folder name ('database' for an image already provided, 'images' for a new image )
	\t\t image name
	\t\t semantic file name

	To perform task 7:
	\tpython ./main.py t7

	\tFollowed by providing:
	\t\t local folder name ('database' for an image already provided, 'images' for a new image )
	\t\t image name
	\t\t semantic file name

	To perform task 8:
	\tpython ./main.py t8

	\tFollowed by providing:
	\t\t subject subject similarity matrix file name
	\t\t specified n value
	\t\t specified m value


	Our group lost a member so if you are trying to implement task 9, LDA, or K-means you will not be able to
	as those parts of the project were taken away to help adjust for our missing member.



	'''

	print(message)

	

def fullTest_Task1_Task2():
	types = [31]

	models = ["HOG", "ELBP", "CM"]
	k = 50
	decomps = ["PCA","SVD"]

	for index, t in enumerate(types):
		for model in models:
			for decomp in decomps:
				get_Subject_or_Type_Weight_Pairs(model, t, k, decomp, 1)



if __name__ == '__main__':
	main()