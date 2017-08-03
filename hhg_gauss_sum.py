from sdm_schrodinger1D import SDMSchrodinger1D, np, ne
from y_gauss_sum import YGaussSum
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # enable log color plot
from scipy.signal import blackman
from scipy.signal import fftconvolve

__doc__ = """
HHG for integer factorization
"""


def FT(A):
    """
    Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    A = np.array(A)
    minus_one = (-1) ** np.arange(A.size)
    return minus_one * np.fft.fft(minus_one * A)

params = dict(
    # integer to factorize
    N=3 * 7,

    # parameters of Gauss sum
    M=10,
    L=10,

    # carrier frequency
    omega0=0.06,

    # transform limited pulse
    TL_laser_filed="F * sin(omega0 * t) * sin(pi * t / t_final) ** 2",

    F=0.04,
    # the final time of propagation (= 7 periods of laser oscillations)
    t_final=8 * 2. * np.pi / 0.06,

    # parameters of propagator
    V="-1. / sqrt(X ** 2 + 1.37)",
    diff_V="X * (X ** 2 + 1.37) ** (-1.5)",

    pi=np.pi,

    abs_boundary="sin(0.5 * pi * (X + X_amplitude) / X_amplitude) ** (0.05 * dt)",

    X_gridDIM=2 * 1024,
    X_amplitude=140.,

    dt=0.02,
)

########################################################################################################
#
#   Get HHG from the transform limited pulse
#
########################################################################################################

qsys = SDMSchrodinger1D(**params).set_groundstate()
# qsys.propagate_TL(int(round(qsys.t_final / qsys.dt)))

iterations = 750
steps = int(round(qsys.t_final / qsys.dt / iterations))

# save the density for subsequent visualization
propagate_TL_density = [
    np.abs(qsys.propagate_TL(steps)) ** 2 for _ in range(iterations)
]


########################################################################################################

# Plot the High Harmonic Generation spectrum
N = len(qsys.P_average_RHS)
k = np.arange(N)

# frequency range in units of omega0
omega = (k - N / 2) * np.pi / (0.5 * qsys.t) / qsys.omega0

# Construct the target as a modulated transform limited HHG
TL_signal = blackman(N) * qsys.X_average
ygs = YGaussSum(**params)

#Y = fftconvolve(ygs() * blackman(N), TL_signal, mode='same') * blackman(N)
# normalization
#Y *= np.linalg.norm(TL_signal, np.inf) / np.linalg.norm(Y, np.inf)
Y = TL_signal
#
# # spectra of the HHG
# spectra = np.abs(FT(TL_signal)) ** 2
# plt.semilogy(
#     omega,
#     #spectra / spectra.max(),
#     spectra,
#     label='Transformed Limited'
# )
#
# spectra = np.abs(FT(Y)) ** 2
# plt.semilogy(
#     omega,
#     #spectra / spectra.max(),
#     spectra,
#     label='Target'
# )
#
# plt.legend()
# plt.ylabel('spectrum (arbitrary units)')
# plt.xlabel('frequency / $\\omega0$')
# plt.xlim([0, 45.])
# #plt.ylim([1e-15, 1.])
#
#
# plt.show()

# initialize tracking
tracking = SDMSchrodinger1D(Y=Y, **params)

# set initial condition to be the ground state
tracking.set_groundstate()

########################################################################################################
#
#   Propagate
#
########################################################################################################

plt.subplot(121)
plt.title("Propagation target")

plt.imshow(
    propagate_TL_density,
    origin='lower',
    norm=LogNorm(vmin=1e-12, vmax=0.1),
    aspect=0.4, # image aspect ratio
    extent=[qsys.X.min(), qsys.X.max(), 0., qsys.t]
)
plt.xlabel('coordinate $x$ (a.u.)')
plt.ylabel('time $t$ (a.u.)')

plt.colorbar()

plt.subplot(122)
plt.title("Propagation during tracking")
# display the propagator
plt.imshow(
    [np.abs(tracking.propagate(steps)) ** 2 for _ in range(iterations)],
    origin='lower',
    norm=LogNorm(vmin=1e-12, vmax=0.1),
    aspect=0.4, # image aspect ratio
    extent=[tracking.X.min(), tracking.X.max(), 0., tracking.t]
)
plt.xlabel('coordinate $x$ (a.u.)')
plt.ylabel('time $t$ (a.u.)')

plt.colorbar()

plt.show()

# initial_state = np.abs(tracking.wavefunction) ** 2
# final_state = np.abs(tracking.propagate(tracking.Y.size)) ** 2

########################################################################################################
#
#   Plot
#
########################################################################################################
#
# plt.title("Initial and final states")
#
# plt.semilogy(tracking.X, initial_state, label="initial")
# plt.semilogy(tracking.X, final_state, label="final")
#
# plt.xlabel("$x$ (a.u.)")
# plt.ylabel("denisty")
# plt.legend()
#
# plt.show()

##################################################################################################

plt.subplot(121)
plt.title("the first Ehrenfest theorem (Tracking)")

times = tracking.dt * np.arange(len(tracking.X_average))
plt.plot(
    times,
    np.gradient(tracking.X_average, tracking.dt),
    '-r',
    label='$d\\langle\\hat{x}\\rangle / dt$'
)
plt.plot(times, tracking.X_average_RHS, '--b', label='$\\langle\\hat{p}\\rangle$')
plt.legend()
plt.ylabel('momentum')
plt.xlabel('time $t$ (a.u.)')

plt.subplot(122)
plt.title("the first Ehrenfest theorem (Original)")

plt.plot(
    times,
    np.gradient(qsys.X_average, qsys.dt),
    '-r',
    label='$d\\langle\\hat{x}\\rangle / dt$'
)
plt.plot(times, qsys.X_average_RHS, '--b', label='$\\langle\\hat{p}\\rangle$')
plt.legend()
plt.ylabel('momentum')
plt.xlabel('time $t$ (a.u.)')

plt.show()

plt.subplot(121)
plt.title("the second Ehrenfest theorem (Tracking)")

plt.plot(
    times,
    np.gradient(tracking.P_average, tracking.dt),
    '-r',
    label='$d\\langle\\hat{p}\\rangle / dt$'
)
plt.plot(times, tracking.P_average_RHS, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
plt.legend()
plt.ylabel('force')
plt.xlabel('time $t$ (a.u.)')

plt.subplot(122)
plt.title("the second Ehrenfest theorem (Original)")

plt.plot(
    times,
    np.gradient(qsys.P_average, qsys.dt),
    '-r',
    label='$d\\langle\\hat{p}\\rangle / dt$'
)
plt.plot(times, qsys.P_average_RHS, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
plt.legend()
plt.ylabel('force')
plt.xlabel('time $t$ (a.u.)')

plt.show()


##################################################################################################

plt.title("Laser field and response")

plt.subplot(121)
FT_Y = np.abs(FT(tracking.Y)) ** 2
#FT_Y /= FT_Y.max()
plt.semilogy(omega, FT_Y, label='target')

FT_E = np.abs(FT(tracking.E)) ** 2
#FT_E /= FT_E.max()
plt.semilogy(omega, FT_E, label='laser field')

TL_field = ne.evaluate(params["TL_laser_filed"], local_dict=params, global_dict={'t' : times})
plt.semilogy(omega, np.abs(FT(TL_field)) ** 2, label="Original field")

plt.legend()
plt.xlabel("$\omega / \omega_0$")
plt.ylabel("$abs(FFT(\\cdot))$")
plt.xlim([0,30])

plt.subplot(122)
plt.title("Target and objective")

plt.semilogy(omega, np.abs(FT(tracking.Y)) ** 2, label="Target")
plt.semilogy(omega, np.abs(FT(blackman(N) * tracking.X_average)) ** 2, label="$\\langle X \\rangle$")
plt.xlim([0,30])
plt.legend()
plt.xlabel("$\omega / \omega_0$")

plt.show()

##################################################################################################

plt.subplot(121)
plt.title("Target vs Objective")
plt.plot(times, tracking.Y, label="Target")
plt.plot(times, tracking.X_average, label="$\\langle X \\rangle$")
plt.xlabel("time ($t$)")
plt.legend()


plt.subplot(122)
plt.title("Original vs Tracking")
plt.plot(times, tracking.E, label="tracking field")
plt.plot(times, TL_field, label="Original field")
plt.xlabel("time ($t$)")
plt.legend()

plt.show()