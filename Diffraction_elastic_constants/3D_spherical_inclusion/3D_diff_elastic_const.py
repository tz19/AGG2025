import numpy as np
import matplotlib.pyplot as plt


def grain_compliance_tensor(C11, C12, C44):
    """
    MATLAB function U(C11,C12,C44) translated to Python.
    Returns (a, b, c).
    """
    # Bulk modulus K and shear modulus G for a polycrystal
    K = (C11 + 2.0 * C12) / 3.0

    g_roots = np.roots(
        [
            8.0,
            5.0 * C11 + 4.0 * C12,
            -C44 * (7.0 * C11 - 4.0 * C12),
            -C44 * (C11 - C12) * (C11 + 2.0 * C12),
        ]
    )
    # MATLAB: G = G_roots(2) (1-based index) -> Python index 1
    G = np.real_if_close(g_roots[1]).item()

    # Eshelby solution and constrained compliance tensor
    gamma = K / (3.0 * K + 4.0 * G)
    delta = 3.0 * (K + 2.0 * G) / (5.0 * (3.0 * K + 4.0 * G))
    L0 = [3.0 * K, 2.0 * G, 2.0 * G]
    M0 = [x ** -1 for x in L0]
    L1 = [C11 + 2.0 * C12, C11 - C12, 2.0 * C44]
    S = [3.0 * gamma, 2.0 * delta, 2.0 * delta]
    I = [1.0, 1.0, 1.0]
    T = [(I[i] + (S[i] * M0[i]) * (L1[i] - L0[i])) ** -1 for i in range(3)]
    U = [T[i] * M0[i] for i in range(3)]
    a = U[0] / 3.0
    b = U[1] / 2.0
    c = U[2] / 2.0
    return a, b, c


def orientation_index(h, k, l):
    return (h**2 * k**2 + l**2 * k**2 + h**2 * l**2) / (h**2 + k**2 + l**2) ** 2


def Gamma_to_n(Gamma):
    """
    Generate one direction n = [h, k, l] for a target orientation index gamma.

    Derivation used:
    1) Let x = h^2/R, y = k^2/R, z = l^2/R, where R = h^2 + k^2 + l^2.
       Then x, y, z >= 0, x + y + z = 1, and gamma = xy + yz + zx.
    2) Choose a one-parameter path x = y = t, z = 1 - 2t, t in [0, 1/2].
       Along this path, gamma(t) = 2t - 3t^2.
    3) Invert gamma(t): t = (1 - sqrt(1 - 3*gamma)) / 3, for gamma in [0, 1/3].
    4) Recover one valid direction:
       n = [sqrt(t), sqrt(t), sqrt(1 - 2t)].
    """
    if Gamma < 0.0 or Gamma > 1.0 / 3.0:
        raise ValueError("gamma must be in [0, 1/3].")

    t = (1.0 - np.sqrt(1.0 - 3.0 * Gamma)) / 3.0
    h = np.sqrt(t)
    k = np.sqrt(t)
    l = np.sqrt(1.0 - 2.0 * t)
    return np.array([h, k, l], dtype=float)

def symbolic2full(symbolic):
    d_ijkl = np.zeros((3, 3, 3, 3))
    d_ijkl[0, 0, 0, 0] = 1
    d_ijkl[1, 1, 1, 1] = 1
    d_ijkl[2, 2, 2, 2] = 1
    I = np.eye(3)
    full1 = (symbolic[0] - symbolic[1]) * (1.0 / 3.0) * np.einsum('ij,kl->ijkl', I, I)
    full2 = symbolic[2] * (1.0 / 2.0) * (np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I))
    full3 = (symbolic[1] - symbolic[2]) * d_ijkl
    full = full1 + full2 + full3
    return full


def diffraction_elastic_const_3d(n, C11, C12, C44):
    """
    MATLAB function diffractionElasticConst3D(n,C11,C12,C44) translated to Python.
    n: iterable with [h, k, l]
    """
    a, b, c = grain_compliance_tensor(C11, C12, C44)
    h, k, l = n[0], n[1], n[2]
    gamma = orientation_index(h, k, l)
    S = (3.0 * a + 4.0 * b) / 3.0 - 4.0 * (b - c) * gamma
    E = 1.0 / S
    return E

def critical_area(sigma_y, sigma_0, k_y):
    # The Hall-Petch relationship for the critical area
    # sigma_y = sigma_0 + k_y * A^(-1/4)
    A_critical = ( (sigma_y - sigma_0) / k_y ) ** (-4)
    return A_critical



if __name__ == "__main__":
    # Example: material parameters for Ni
    C11 = 246.5e9
    C12 = 147.3e9
    C44 = 124.7e9
    
    sigma_0 = 1.0e9 # Pa
    k_y = 400e3 # Pa m^(1/2)
    
    sigma_applied = 1.4e9
    
    [a, b, c] = grain_compliance_tensor(C11, C12, C44)
    U_full = symbolic2full([3*a, 2*b, 2*c])
    L = symbolic2full([C11+2*C12, C11-C12, 2*C44])
    
    gamma_vals = np.linspace(0, 1.0 / 3.0, 100)
    E = np.zeros_like(gamma_vals)
    max_principal_stresses = np.zeros_like(gamma_vals)
    Areas = np.zeros_like(gamma_vals)
    for i, gamma_val in enumerate(gamma_vals):
        n = Gamma_to_n(gamma_val)
        n = n/np.linalg.norm(n)
        # Calculate the diffraction elastic constant along the direction n (loading direction)
        E[i] = diffraction_elastic_const_3d(n, C11, C12, C44)
        
        # Everything below is in the crystal coordinate system
        # The macroscopic stress tensor 
        stress_bar = np.einsum('i,j->ij', n, n) * sigma_applied
        strain = np.einsum('ijkl,kl->ij', U_full, stress_bar)
        stress = np.einsum('ijkl,kl->ij', L, strain)
        # The maximum principal stress is the largest eigenvalue of the stress tensor
        max_principal_stresses[i] = np.linalg.eig(stress)[0].max()
        Areas[i] = critical_area(max_principal_stresses[i], sigma_0, k_y)

    
    plt.plot(Areas*1e12, E/1e9, '-r', linewidth=2.0)
    plt.xscale('log')
    plt.xlim(1e-2, 1e2)
    plt.xlabel('Critical Area (um^2)')
    plt.ylabel('E_LD (GPa)')
    plt.savefig('Diffraction_elastic_const_3d_critical_area.png', dpi=300)
    
    
    