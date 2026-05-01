import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  Tensor builders (full 4th-order, no symbolic shortcut)
# ---------------------------------------------------------------------------

def identity_4th():
    """Symmetric identity on 2nd-order symmetric tensors:
        I_ijkl = (1/2) (delta_ik delta_jl + delta_il delta_jk).
    """
    I3 = np.eye(3)
    return 0.5 * (np.einsum('ik,jl->ijkl', I3, I3)
                  + np.einsum('il,jk->ijkl', I3, I3))


def isotropic_stiffness_tensor(K, G):
    """Full 4th-order isotropic stiffness tensor with bulk modulus ``K`` and
    shear modulus ``G``:
        C_ijkl = (K - 2G/3) delta_ij delta_kl + G (delta_ik delta_jl + delta_il delta_jk).
    """
    I3 = np.eye(3)
    return ((K - 2.0 * G / 3.0) * np.einsum('ij,kl->ijkl', I3, I3)
            + G * (np.einsum('ik,jl->ijkl', I3, I3)
                   + np.einsum('il,jk->ijkl', I3, I3)))


def cubic_stiffness_tensor(C11, C12, C44):
    """Full 4th-order stiffness tensor of a cubic single crystal expressed in
    its own crystal-aligned frame.  Built directly from the standard cubic
    decomposition (no use of any "symbolic" 3-component form):

        C_ijkl = C12 delta_ij delta_kl
               + C44 (delta_ik delta_jl + delta_il delta_jk)
               + (C11 - C12 - 2 C44) sum_a (e_a otimes e_a otimes e_a otimes e_a),

    where the last term carries the cubic anisotropy (a = 1, 2, 3 are the
    crystal axes).
    """
    I3 = np.eye(3)
    d = np.zeros((3, 3, 3, 3))
    for a in range(3):
        d[a, a, a, a] = 1.0
    return (C12 * np.einsum('ij,kl->ijkl', I3, I3)
            + C44 * (np.einsum('ik,jl->ijkl', I3, I3)
                     + np.einsum('il,jk->ijkl', I3, I3))
            + (C11 - C12 - 2.0 * C44) * d)


def eshelby_cylinder(nu):
    """Eshelby tensor of an infinite cylindrical inclusion whose axis is along
    the lab z-axis, embedded in an isotropic matrix of Poisson ratio ``nu``
    (Mura, *Micromechanics of Defects in Solids*, eq. 11.22).

    The tensor is anisotropic (z is special), so it is *not* expressible in
    the [s1, s2, s3] symbolic form used for the spherical case.
    """
    S = np.zeros((3, 3, 3, 3))
    fac = 1.0 / (2.0 * (1.0 - nu))

    S[0, 0, 0, 0] = fac * (3.0 / 4.0 + (1.0 - 2.0 * nu) * 0.5)
    S[1, 1, 1, 1] = S[0, 0, 0, 0]

    S[0, 0, 1, 1] = fac * (1.0 / 4.0 - (1.0 - 2.0 * nu) * 0.5)
    S[1, 1, 0, 0] = S[0, 0, 1, 1]

    S[0, 0, 2, 2] = fac * nu
    S[1, 1, 2, 2] = fac * nu

    S1212 = fac * (1.0 / 4.0 + (1.0 - 2.0 * nu) * 0.5)
    S[0, 1, 0, 1] = S1212
    S[1, 0, 0, 1] = S1212
    S[0, 1, 1, 0] = S1212
    S[1, 0, 1, 0] = S1212

    for (a, b) in ((1, 2), (2, 0)):
        val = 0.25
        S[a, b, a, b] = val
        S[b, a, a, b] = val
        S[a, b, b, a] = val
        S[b, a, b, a] = val

    return S


# ---------------------------------------------------------------------------
#  4th-order tensor algebra helpers
# ---------------------------------------------------------------------------

def tensor_inv4(L):
    """Invert a minor-symmetric 4th-order tensor.

    Reshape the (3,3,3,3) array to a (9,9) matrix, take a pseudo-inverse, and
    reshape back.  This produces a tensor T such that T : L = I (the symmetric
    identity) on the subspace of symmetric 2nd-order tensors.
    """
    return np.linalg.pinv(L.reshape(9, 9)).reshape(3, 3, 3, 3)


def tensor_dot4(A, B):
    """Double-contraction of two 4th-order tensors: (A : B)_ijmn = A_ijkl B_klmn."""
    return np.einsum('ijkl,klmn->ijmn', A, B)


def rotate_4th(L, Q):
    """Rotate a 4th-order tensor: L'_ijkl = Q_ip Q_jq Q_kr Q_ls L_pqrs."""
    return np.einsum('ip,jq,kr,ls,pqrs->ijkl', Q, Q, Q, Q, L)


def zxz_rotation_matrix(eul):
    """General ZXZ (Bunge) Euler rotation matrix, with angles in degrees.

    Parameters
    ----------
    eul : array-like of length 3
        ``(alpha, beta, gamma)`` in degrees.  The convention follows
        ``zxz_rotation_matrix.m``:

            R = Rz(gamma) @ Rx(beta) @ Rz(alpha),

        i.e. ``e_global = R @ e_local``.

    Notes
    -----
    Setting ``eul = (theta, 0, 0)`` reduces to a pure rotation about the
    cylinder axis (lab z) by ``theta``.
    """
    alpha, beta, gamma = (np.deg2rad(a) for a in eul)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rz_a = np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]])
    Rx_b = np.array([[1.0, 0.0, 0.0],
                     [0.0,  cb, -sb],
                     [0.0,  sb,  cb]])
    Rz_g = np.array([[cg, -sg, 0.0],
                     [sg,  cg, 0.0],
                     [0.0, 0.0, 1.0]])
    return Rz_g @ Rx_b @ Rz_a


def Q_from_directions(n_crystal, c_crystal):
    """Build a proper rotation matrix Q (e_lab = Q @ e_crystal) such that the
    lab x-axis is the crystal direction ``n_crystal`` (= loading direction) and
    the lab z-axis is the crystal direction ``c_crystal`` (= cylinder axis).

    Both inputs are unit-normalised internally; ``c_crystal`` is also
    re-orthogonalised against ``n_crystal`` to be safe.
    """
    n_c = np.asarray(n_crystal, dtype=float)
    n_c = n_c / np.linalg.norm(n_c)
    c_c = np.asarray(c_crystal, dtype=float)
    c_c = c_c - np.dot(c_c, n_c) * n_c
    c_c = c_c / np.linalg.norm(c_c)
    e_y = np.cross(c_c, n_c)        # crystal direction along lab y
    # rows of Q are the crystal-frame components of the lab basis vectors
    return np.vstack([n_c, e_y, c_c])


# ---------------------------------------------------------------------------
#  Polycrystal effective moduli (Eshelby self-consistent for cubic)
# ---------------------------------------------------------------------------

def polycrystal_moduli(C11, C12, C44):
    """Self-consistent effective bulk and shear moduli of an untextured cubic
    polycrystal.  The bulk modulus is exact (Hill); the shear modulus is the
    relevant root of the standard self-consistent quartic.
    """
    K = (C11 + 2.0 * C12) / 3.0
    G_roots = np.roots([8.0,
                        5.0 * C11 + 4.0 * C12,
                        -C44 * (7.0 * C11 - 4.0 * C12),
                        -C44 * (C11 - C12) * (C11 + 2.0 * C12)])
    G = float(np.real_if_close(G_roots[1]))
    return K, G


# ---------------------------------------------------------------------------
#  Diffraction elastic constant in 2D (cylindrical grain)
# ---------------------------------------------------------------------------

def diffraction_elastic_const_2d(rot, n_lab, C11, C12, C44):
    """Diffraction (single-grain) elastic modulus along the lab-frame loading
    direction ``n_lab`` for a cylindrical grain whose crystal-to-lab rotation is
    given by ``rot``.

    ``rot`` may be either
      - an Euler triple ``(alpha, beta, gamma)`` in degrees
        (ZXZ / Bunge convention, see :func:`zxz_rotation_matrix`), or
      - a 3x3 proper rotation matrix Q with the convention
        ``e_lab = Q @ e_crystal``  (built e.g. by :func:`Q_from_directions`).

    Returns
    -------
    E      : float       diffraction elastic modulus along ``n_lab`` (Pa)
    U_lab  : (3,3,3,3)   constrained compliance in the lab frame
    L_lab  : (3,3,3,3)   crystal stiffness rotated into the lab frame
    """

    # all tensors in the lab frame.
    K_bar, G_bar = polycrystal_moduli(C11, C12, C44)
    nu_bar = (3.0 * K_bar - 2.0 * G_bar) / (2.0 * (3.0 * K_bar + G_bar))

    L_bar = isotropic_stiffness_tensor(K_bar, G_bar)
    M_bar = tensor_inv4(L_bar)
    L0    = cubic_stiffness_tensor(C11, C12, C44)

    Q = np.asarray(rot, dtype=float)
    if Q.shape == (3,):
        Q = zxz_rotation_matrix(Q)
    elif Q.shape != (3, 3):
        raise ValueError("rot must be Euler triple (3,) or rotation matrix (3,3)")
    L_lab = rotate_4th(L0, Q)

    S_esh = eshelby_cylinder(nu_bar)
    I4    = identity_4th()

    # U = (I + S : M_bar : (L - L_bar))^{-1} : M_bar
    A = I4 + tensor_dot4(S_esh, tensor_dot4(M_bar, L_lab - L_bar))
    T = tensor_inv4(A)
    U_lab = tensor_dot4(T, M_bar)

    s_n = np.einsum('ijkl,i,j,k,l->', U_lab, n_lab, n_lab, n_lab, n_lab)
    return 1.0 / s_n, U_lab, L_lab


def critical_area(sigma_y, sigma_0, k_y):
    """Hall-Petch critical area: sigma_y = sigma_0 + k_y * A^(-1/4)."""
    return ((sigma_y - sigma_0) / k_y) ** (-4)


# ---------------------------------------------------------------------------
#  Main: phase boundary (critical area vs E_LD) for a cylindrical grain
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Single-crystal elastic constants (Ni), same as the 3D example
    C11 = 246.5e9   # Pa
    C12 = 147.3e9   # Pa
    C44 = 124.7e9   # Pa

    # Hall-Petch parameters
    sigma_0 = 1.0e9     # Pa
    k_y     = 400.0e3   # Pa * m^(1/2)

    # Macroscopic uniaxial stress
    sigma_applied = 1.4e9   # Pa

    # Loading direction in the lab frame (perpendicular to the cylinder axis)
    n = np.array([1.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    #  Sweep over orientation index gamma in [0, 1/3]
    # ------------------------------------------------------------------
    # In 2D the diffraction modulus depends on TWO things: the loading
    # direction in the crystal (which fixes gamma) and the cylinder-axis
    # orientation in the crystal (perpendicular to the loading direction; one
    # extra DoF).  For each gamma we therefore scan over the cylinder axis
    # orientation psi and report both the lower and upper envelopes of E_LD.
    #
    # Loading path n_crystal(t) = [cos t, sin t/sqrt(2), sin t/sqrt(2)] traces
    # from <100> (t=0, gamma=0) to <111>/sqrt(3) (t=arccos(1/sqrt(3)),
    # gamma=1/3); on this path the orientation index is
    #     gamma(t) = sin^2 t - (3/4) sin^4 t.
    gamma_vals = np.linspace(0.0, 1.0 / 3.0, 200)
    sin2_t     = (2.0 / 3.0) * (1.0 - np.sqrt(1.0 - 3.0 * gamma_vals))
    sin_t      = np.sqrt(sin2_t)
    cos_t      = np.sqrt(1.0 - sin2_t)

    psi_grid   = np.linspace(0.0, np.pi, 91, endpoint=False)   # cyl-axis sweep

    E_min = np.zeros_like(gamma_vals)
    E_max = np.zeros_like(gamma_vals)
    A_min = np.zeros_like(gamma_vals)   # critical area on the lower-E envelope
    A_max = np.zeros_like(gamma_vals)   # critical area on the upper-E envelope

    for i, (st, ct) in enumerate(zip(sin_t, cos_t)):
        n_cr = np.array([ct, st / np.sqrt(2.0), st / np.sqrt(2.0)])
        # Orthonormal basis of the plane perpendicular to n_cr in the crystal:
        # e1 is independent of t (always perpendicular to this n_cr family),
        # e2 is the in-plane partner.
        e1 = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)
        e2 = np.array([-st, ct / np.sqrt(2.0), ct / np.sqrt(2.0)])

        E_grid = np.empty_like(psi_grid)
        A_grid = np.empty_like(psi_grid)
        for j, psi in enumerate(psi_grid):
            c_cr = np.cos(psi) * e1 + np.sin(psi) * e2     # cylinder axis (crystal)
            Q    = Q_from_directions(n_cr, c_cr)
            E_j, U_lab, L_lab = diffraction_elastic_const_2d(Q, n, C11, C12, C44)

            stress_bar = sigma_applied * np.outer(n, n)
            eps_grain  = np.einsum('ijkl,kl->ij', U_lab, stress_bar)
            sig_grain  = np.einsum('ijkl,kl->ij', L_lab, eps_grain)
            sig_grain  = 0.5 * (sig_grain + sig_grain.T)

            sigma_max  = np.linalg.eigvalsh(sig_grain).max()
            E_grid[j]  = E_j
            A_grid[j]  = critical_area(sigma_max, sigma_0, k_y)

        j_lo, j_hi = E_grid.argmin(), E_grid.argmax()
        E_min[i], A_min[i] = E_grid[j_lo], A_grid[j_lo]
        E_max[i], A_max[i] = E_grid[j_hi], A_grid[j_hi]

    # ------------------------------------------------------------------
    #  Phase boundary plot (mirrors the 3D figure) - now a band
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.fill(np.concatenate([A_min, A_max[::-1]]) * 1e12,
            np.concatenate([E_min, E_max[::-1]]) / 1e9,
            color='red', alpha=0.20, label='cyl-axis envelope')
    ax.plot(A_min * 1e12, E_min / 1e9, '-r', linewidth=2.0,
            label='lower envelope (most compliant cyl axis)')
    ax.plot(A_max * 1e12, E_max / 1e9, '--r', linewidth=1.5,
            label='upper envelope (stiffest cyl axis)')
    ax.set_xscale('log')
    ax.set_xlim(1e-2, 1e2)
    ax.set_xlabel(r'Critical Area ($\mu m^2$)')
    ax.set_ylabel(r'$E_{\mathrm{LD}}$ (GPa)')
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig('Diffraction_elastic_const_2d_critical_area.png', dpi=300)

    # Companion plot: E_LD vs orientation index gamma (matches 3D x-axis)
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.8))
    ax2.fill_between(gamma_vals, E_min / 1e9, E_max / 1e9,
                     color='red', alpha=0.20, label='cyl-axis envelope')
    ax2.plot(gamma_vals, E_min / 1e9, '-r', linewidth=2.0,
             label='lower envelope')
    ax2.plot(gamma_vals, E_max / 1e9, '--r', linewidth=1.5,
             label='upper envelope')
    ax2.set_xlabel(r'orientation index $\gamma$')
    ax2.set_ylabel(r'$E_{\mathrm{LD}}$ (GPa)')
    ax2.set_xlim(0.0, 1.0 / 3.0)
    ax2.grid(True, alpha=0.4)
    ax2.legend(loc='upper left', fontsize=9)
    fig2.tight_layout()
    fig2.savefig('Diffraction_elastic_const_2d_E_vs_gamma.png', dpi=300)

    print(f"E_LD lower envelope: [{E_min.min()/1e9:.2f}, {E_min.max()/1e9:.2f}] GPa")
    print(f"E_LD upper envelope: [{E_max.min()/1e9:.2f}, {E_max.max()/1e9:.2f}] GPa")
    print(f"Critical area (lower-E envelope): "
          f"[{A_min.min()*1e12:.3e}, {A_min.max()*1e12:.3e}] um^2")
    print("Saved: Diffraction_elastic_const_2d_critical_area.png")
    print("Saved: Diffraction_elastic_const_2d_E_vs_gamma.png")
