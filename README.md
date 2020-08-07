# Eigenfaces
Implementing a Python version of the facial analysis program we demonstrated in lecture, using Principal Components Analysis (PCA).

Dataset: YaleB_32x32.mat

load_and_center_dataset(filename) — load the dataset from a provided .mat file, re-center it around the origin and return it as a NumPy array of floats
get_covariance(dataset) — calculate and return the covariance matrix of the dataset as a NumPy matrix (d x d array)
get_eig(S, m) — perform eigen decomposition on the covariance matrix S and return a diagonal matrix (NumPy array) with the largest m eigenvalues on the diagonal, and a matrix (NumPy array) with the corresponding eigenvectors as columns
project_image(image, U) — project each image into your m-dimensional space and return the new representation as a d x 1 NumPy array
display_image(orig, proj) — use matplotlib to display a visual representation of the original image and the projected image side-by-side


