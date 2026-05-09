import numpy as np 
from scipy.linalg import eig
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors

# a   = 2.0    # source term for w
m   = 0.45      # decay rate for n
v   = 182.5      # advection velocity for w (can be negative)

k_array = np.linspace(0.0, 10, 200)
a_range = np.linspace(2*m, 2.5, 100)
eig_values = np.zeros((len(a_range),(len(k_array))))

for j, a in enumerate(a_range): 

    # if a < 2*m:
    #     eig_values[j, :] = np.nan   # or 0.0 if you prefer
    #     continue

    n_eq = (a + np.sqrt((a + 2.*m)*(a -2.*m)))/(2.*m)
    w_eq = a / (1.0 + (n_eq*n_eq))
    # w_eq = a 
    # n_eq = 0.0 

    for i, k in enumerate(k_array): 

        J = np.array([[(-(1 + n_eq**2) + v * k * 1j), (-2 * w_eq * n_eq)], 
                    [(n_eq**2), (2 * w_eq * n_eq - m - k**2)]])
        
        # Solve eigenvalues: 
        eigvals, _,  = eig(J) 
        max_eigval = np.max(np.real(eigvals))

        # eig_values[j, i] = max_eigval 

        if max_eigval < 0: 
            eig_values[j, i] = 0 
        elif max_eigval > 0: 
            eig_values[j, i] = 1

        
# If you only want to see 
fig, ax = plt.subplots() 
k_coords, a_coords  = np.meshgrid(k_array, a_range)
pcm = ax.pcolormesh(k_coords, a_coords, eig_values)
pcm.set_clim(0,1)

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label('Positive Real Eigenvalue Existence', fontsize=12)

ax.set_xlabel('k', fontsize=12)
ax.set_ylabel('a', fontsize=12)
ax.set_title(f'Stability diagram, {m =}, {v = }', fontsize=13)

# plt.savefig("eigenvalues.png", transparent = True)

# If you want to see all eigenvalues 

# fig, ax = plt.subplots() 
# k_coords, a_coords  = np.meshgrid(k_array, a_range)
# pcm = ax.pcolormesh(k_coords, a_coords, eig_values)

# cbar = fig.colorbar(pcm, ax=ax)
# cbar.set_label('Max real eigenvalue', fontsize=12)

# ax.set_xlabel('k', fontsize=12)
# ax.set_ylabel('a', fontsize=12)
# ax.set_title('Stability diagram', fontsize=13)
plt.show()




