To get a look at the help message, run:
python ./main.py help

To perform task 1:
python ./main.py t1
Followed by providing:
feature model (CM, ELBP, HOG)
specified X value (cc, con, detail, emboss, jitter, neg, noise1, noise2, original, poster, rot, smooth, stipple)
specified k value (1-192 for CM, 1-32 for ELBP, 1-1764 for HOG)
specified dimensionality reduction technique (SVD, PCA)

Two files will be created in the latent semantic folder:
latent_semantics_(dimensional_reduction_technique)_(feature_model)_(type)_(k).txt 
which has the subject-weight pairs and
rlatent_semantics_(dimensional_reduction_technique)_(feature_model)_(type)_(k).txt 
which has k by f matrix for projection.



To perform task 2:
python ./main.py t2
Followed by providing:
feature model (CM, ELBP, HOG)
specified Y value (1-40)
specified k value (1-192 for CM, 1-32 for ELBP, 1-1764 for HOG)
specified dimensionality reduction technique (SVD, PCA)

Two files will be created in the latent semantic folder:
latent_semantics_(dimensional_reduction_technique)_(feature_model)_(subject_number)_(k).txt 
which has the subject-weight pairs and
rlatent_semantics_(dimensional_reduction_technique)_(feature_model)_(subject_number)_(k).txt 
which has k by f matrix for projection.



To perform task 3:
python ./main.py t3
Followed by providing:
feature model (CM, ELBP, HOG)
specified k value (1-12)
specified dimensionality reduction technique (SVD, PCA)
input directory (database)

Two files will be created:
latent_semantics_(type_type)_{dimentionality_reduction}_{feature_model}_{k}.txt 
which will have the type-weight pairs and is stored in the latent semantic folder and 
(type_type)_{feature_model}.txt which has the similarity matrix for the types and is stored in the sim_matrix folder.



To perform task 4:
python ./main.py t4
Followed by providing:
feature model (CM, ELBP, HOG)
specified k value (1-40)
specified dimensionality reduction technique (SVD, PCA)
input directory (database)

Two files will be created:
latent_semantics_(subject_subject)_{dimentionality_reduction}_{feature_model}_{k}.txt 
which will have the type-weight pairs and is stored in the latent semantic folder and 
(subject_subject)_{feature_model}.txt which has the similarity matrix for the types and is stored in the sim_matrix folder.

Just use database as the directory name to avoid any issues.



To perform task 5:
python ./main.py t5
Followed by providing:
folder name (images,database)
file name 
latent semantic file name
number of images to return

Place a new image in images and then use that directory for the folder name. If using an image in the database use database as the image name.
Image names and latent semantic files should include the .png and .txt file endings.
For the latent semantic files just use the name without the r in front. For example use
latent_semantics_PCA_CM_cc_20.txt even though the real file will be that with an r in front. These latent semantic files should be files generated from tasks 1 and 2.



To perform task 6:
python ./main.py t6
Followed by providing:
folder name (images,database)
file name 
latent semantic file name

Place a new image in images and then use that directory for the folder name. If using an image in the database use database as the image name.
Image names and latent semantic files should include the .png and .txt file endings.
For the latent semantic files just use the name without the r in front. For example use
latent_semantics_PCA_CM_cc_20.txt even though the real file will be that with an r in front. These latent semantic files should be files generated from tasks 1 and 2.



To perform task 7:
python ./main.py t7
Followed by providing:
folder name (images,database)
file name 
latent semantic file name

Place a new image in images and then use that directory for the folder name. If using an image in the database use database as the image name.
Image names and latent semantic files should include the .png and .txt file endings.
For the latent semantic files just use the name without the r in front. For example use
latent_semantics_PCA_CM_cc_20.txt even though the real file will be that with an r in front. These latent semantic files should be files generated from tasks 1 and 2.



To perform task 8:
python ./main.py t8
Followed by providing
matrix file name
n value
mm value

The matrix file name should come from the sim_matrix folder and should include the file ending.
