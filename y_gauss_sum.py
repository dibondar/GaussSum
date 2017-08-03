import numpy as np
import numexpr as ne
from types import MethodType, FunctionType


class YGaussSum:
    """
    Calculate the Gauss sum [Y(t) in the notes] to be used as the target in tracking.
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            dt -- time step
            omega0 -- the carrier frequency
            t_final -- time = [0, t_final]

            M, L -- (positive integers) needed to calculate the Gaussian sum
            N -- the integer to be factorized
        """

        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t_final
        except AttributeError:
            raise AttributeError("Final time point (t_final) was not specified")

        try:
            self.omega0
        except AttributeError:
            raise AttributeError("Carrier frequency (omega0) was not specified")

        try:
            self.M
        except AttributeError:
            raise AttributeError("Positive integer (M) was not specified")

        try:
            self.L
        except AttributeError:
            raise AttributeError("Positive integer (L) was not specified")

        # initialize the time axis
        self.t = np.arange(0, self.t_final, self.dt)

    def get_gauss_sum(self):
        """
        Return the Gauss sum. A_N^{(M)}(l) in the notes.
        :return:
        """
        m = np.arange(0, self.M + 1)
        m = m[np.newaxis, :]

        l = np.arange(1, self.L + 1)
        l = l[:, np.newaxis]
        pi = np.pi

        result = ne.evaluate("sum(exp(-2j * pi * m ** 2 * N / l) / (M + 1), axis=1)", global_dict=self.__dict__)
        return result

    def __call__(self):
        """
        Calculate the tracking target Y(t)
        :return:
        """
        A = self.get_gauss_sum()
        A = A.real[np.newaxis, :]

        l = np.arange(1, self.L + 1)
        l = l[np.newaxis, :]

        t = self.t[:, np.newaxis]

        result = ne.evaluate("sum(A * exp(-1j * l * omega0 * t), axis=1)", global_dict=self.__dict__)
        return result.real

if __name__ == '__main__':

    # Plotting facility
    import matplotlib.pyplot as plt

    hhg_gauss_gauss = YGaussSum(
        dt=0.1,
        omega0=0.1,
        t_final=4 * 2. * np.pi / 0.1,
        L=10,
        M=5,
        N=7*2,
    )

    plt.subplot(121)
    plt.title("Target $Y(t)$")

    print("Numer of time points ", hhg_gauss_gauss.t.size)
    Y = hhg_gauss_gauss()
    plt.plot(hhg_gauss_gauss.t, Y)
    plt.xlabel("time, $t$")

    plt.subplot(122)
    plt.title("Fourier transform of $Y(t)$")

    ################## Calculate the Fourier transform ##################

    k = np.arange(Y.size)
    minus_one = (-1) ** k
    FT_Y = minus_one * np.fft.fft(minus_one * Y)
    FT_Y *= np.exp(-1j * np.pi * Y.size / 2)
    omega = (k - k.size / 2.) * np.pi / (0.5 * hhg_gauss_gauss.t.max())
    #####################################################################

    plt.plot(omega / hhg_gauss_gauss.omega0, FT_Y.real / FT_Y.real.max(), '*-', label="$FT(Y(t))$")
    gauss_sum = hhg_gauss_gauss.get_gauss_sum()
    plt.plot(np.arange(1, gauss_sum.size + 1), gauss_sum.real, '*', label="Gauss sum")

    plt.legend()

    plt.xlabel("$\omega / \omega_0$")
    plt.xlim([0, 20])

    plt.show()