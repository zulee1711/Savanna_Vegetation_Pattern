import numpy as np 
from scipy.linalg import eig
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors

# a   = 2.0    # source term for w
# m   = 0.01       # decay rate for n
v   = 182.5      # advection velocity for w (can be negative)

k_array = np.linspace(0.0, 10, 200)
a_range = np.linspace(0.06, 3, 200)
m_range = np.linspace(0.01, 0.5, 200)
pattern = np.zeros((len(m_range), len(a_range)))
eig_values = np.zeros((len(a_range),(len(k_array))))

for l, m in enumerate(m_range): 
    print(f"{m = }")
    for j, a in enumerate(a_range): 
        print(f"{a = }")

        if a < 2*m:
            eig_values[j, :] = np.nan   # or 0.0 if you prefer
            continue

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

            if max_eigval > 0: 
                pattern[l, j] = 1
                break 
        else: 
            pattern[l, j] = 0

# If you only want to see 
fig, ax = plt.subplots() 
a_coords, m_coords  = np.meshgrid(a_range, m_range)
pcm = ax.pcolormesh(a_coords, m_coords, pattern)
pcm.set_clim(0,1)

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(f'Pattern Existence', fontsize=12)

ax.set_xlabel('a', fontsize=12)
ax.set_ylabel('m', fontsize=12)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title(f'{v = }', fontsize=12)
plt.savefig("pattern_existence.png", transparent = True)
plt.show()

