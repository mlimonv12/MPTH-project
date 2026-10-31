# MPTH project

## 3d-printed THz DOE for a flat-top profile from a collimated gaussian beam



### Theory fundamentals:

We have an incident collimated gaussian beam, which can be described by the following profile:

\begin{equation}
    U_i(x,y) = A_0\exp \left( -\frac{x^2+y^2}{w_0^2} \right) = A_0 \exp \left( \frac{-r^2}{w_0^2} \right) = U_i(r)
\end{equation}

The incident beam is transmitted through the DOE to be designed, with transfer function $U_{DOE}(r)$. The goal is to generate a flat-top image, uniformly illuminated inside of a radius $R$:

$$
 I(r) = \left\{ \begin{array}{cc}
        1 & r \le R \\
        0 & r > R
    \end{array} \right.
$$

We know that the profile at the image plane will be the Fourier transform of the Beam propagated through the DOE, such that:

$$ I(r) = \mathcal{F}\left\{ U_i \cdot U_{DOE} \right\} $$

Given that the problem has revolution symmetry, however, we can solve for the radial component of the DOE profile using the zero-order Hankel transform:

$$ F_0(r) = \int_0^{\infty}f(r)J_0(r) r \diff{r} $$