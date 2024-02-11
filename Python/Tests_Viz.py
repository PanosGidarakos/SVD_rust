import numpy as np
import time 
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from G_R_svd import clear_svd as g_r_svd
from non_exact_svd import simplified_svd_2d as n_e_svd
if __name__ == '__main__':

    cat = data.chelsea()
    gray_cat = rgb2gray(cat)
    gray_cat = gray_cat[:, :300]
    r_values = [5, 10, 20, 40, 80, 160]

    # Create a figure and subplots with 2 rows
    # Each row has 1 (original) + 6 (for SVD approximations) plots
    fig, axes = plt.subplots(2, len(r_values) + 1, figsize=(15, 6))  # Adjust figure size as needed

    # Compute SVDs
    u_g_r, s_g_r, v_g_r = g_r_svd(gray_cat.copy())
    singularValues_n_e, us_n_e, vs_n_e = n_e_svd(gray_cat.copy())

    # Plot the original grayscale image in the first row
    axes[0, 0].imshow(gray_cat, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Plot the g_r_svd approximations in the first row
    for i, r in enumerate(r_values):
        approximation = u_g_r[:, :r] @ s_g_r[:r, :r] @ v_g_r[:r, :]
        axes[0, i + 1].imshow(approximation, cmap='gray')
        axes[0, i + 1].set_title(f'G_R_svd, k={r}')
        axes[0, i + 1].axis('off')

    # Plot the original grayscale image again in the second row for consistency
    axes[1, 0].imshow(gray_cat, cmap='gray')
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')

    # Plot the n_e_svd approximations in the second row
    for i, r in enumerate(r_values):
        approximation = us_n_e[:, :r] @ np.diag(singularValues_n_e)[:r, :r] @ vs_n_e[:r, :]
        axes[1, i + 1].imshow(approximation, cmap='gray')
        axes[1, i + 1].set_title(f'N_E_svd, k={r}')
        axes[1, i + 1].axis('off')
    plt.tight_layout()
    plt.savefig('Kittens.png')
    plt.show()                         
