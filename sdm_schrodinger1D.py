import numpy as np
import numexpr as ne
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from types import MethodType, FunctionType


class SDMSchrodinger1D:
    """
    Spectral Dynamical Mimicry
        [Phys. Rev. Lett. 118, 083201 (2017)] https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.083201

     H = 0.5 * p ** 2 + V(x) - x E(t).
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - specifying the grid size
            X_amplitude - maximum value of the coordinates
            V - potential energy (as a string to be evaluated by numexpr)
            Y (optional) - tracking target
            diff_V -- the derivative of the potential energy for the Ehrenfest theorem calculations
            t (optional) - initial value of time
            abs_boundary (optional) -- absorbing boundary (as a string to be evaluated by numexpr)
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
            # make sure self.X_gridDIM has a value of power of 2
            assert 2 ** int(np.log2(self.X_gridDIM)) == self.X_gridDIM, \
                "A value of the grid size (X_gridDIM) must be a power of 2"
        except AttributeError:
            raise AttributeError("Grid size (X_gridDIM) was not specified")

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate range (X_amplitude) was not specified")

        try:
            self.V
        except AttributeError:
            raise AttributeError("Potential energy (V) was not specified")

        try:
            self.diff_V
        except AttributeError:
            raise AttributeError("The derivative of potential energy (diff_V) was not specified")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.Y
            self.diff_diff_Y = np.gradient(
                np.gradient(self.Y, self.dt, edge_order=2),
                self.dt,
                edge_order=2
            )
        except AttributeError:
            print("Warning: The tracking objective (Y) was not specified")


        try:
            self.t
        except AttributeError:
            print("Warning: Initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        try:
            self.abs_boundary
        except AttributeError:
            print("Warning: Absorbing boundary (abs_boundary) was not specified, thus it is turned off")
            self.abs_boundary = 1.

        # get coordinate step size
        self.dX = 2. * self.X_amplitude / self.X_gridDIM

        # generate coordinate range
        self.k = np.arange(self.X_gridDIM)
        self.X = (self.k - self.X_gridDIM / 2) * self.dX

        # generate momentum range
        self.P = (self.k - self.X_gridDIM / 2) * (np.pi / self.X_amplitude)

        # allocate the array for wavefunction
        self.wavefunction = np.zeros(self.X.size, dtype=np.complex)

        # Pre-calculate the exponential of the potential without the laser field: (-)**k * exp(-0.5j * dt * V)
        self.expV = ne.evaluate(
            "(%s) * (-1) ** k * exp(-0.5j * dt * (%s))" % (self.abs_boundary, self.V),
            local_dict=self.__dict__
        )

        # Allocate memory for the exponent of the total potential energy
        self.expVTotal = np.empty_like(self.expV)

        # Pre-calculate the exponential of the kinetic energy
        self.expK = np.exp(-0.5j * self.dt * self.P ** 2) #ne.evaluate("exp(-1.j * dt * 0.5 * P ** 2)", local_dict=self.__dict__)

        ######################################################################################
        #
        #  Initialize the first-order Ehrenfest theorem verification (necessary for tracking)
        #
        ######################################################################################

        # Pre-calculate the energy of the core (V) and its derivative
        self._V = ne.evaluate(self.V, local_dict=self.__dict__)
        self._diff_V = ne.evaluate(self.diff_V, local_dict=self.__dict__)

        self._K = 0.5 * self.P ** 2

        # Lists where the expectation values of X and P
        self.X_average = []
        self.P_average = []

        # Lists where the right hand sides of the Ehrenfest theorems for X and P
        self.diff_V_average = []

        # List where the expectation value of the Hamiltonian will be calculated
        self.hamiltonian_average = []

        # Lists where the right hand sides of the Ehrenfest theorems for X and P
        self.X_average_RHS = self.P_average
        self.P_average_RHS = []

        # Allocate array for storing coordinate or momentum density of the wavefunction
        self.density = np.zeros(self.wavefunction.shape, dtype=np.float)

        # Allocate a copy of the wavefunction for storing the wavefunction in the momentum representation
        self.wavefunction_p = np.zeros_like(self.wavefunction)

        # List where the tracking laser field is going to be saved
        self.E = [0.]

        # Booling fla
        self._warning_not_printed = True

    def propagate(self, time_steps=1):
        """
        Time propagate the wave function saved in self.wavefunction
        :param time_steps: number of self.dt time increments to make
        :return: self.wavefunction
        """
        for _ in range(time_steps):

            # calculate the Ehrenfest theorems
            self.get_Ehrenfest()

            # Find forward derivative of the tracking target
            try:
                diff_Y = (self.Y[len(self.E)] - self.Y[len(self.E) - 1]) / self.dt
            except IndexError:
                print("Tracking completed: Nothing to track in (Y).")
                break

            if self._warning_not_printed and not np.allclose(self.P_average[-1], diff_Y, rtol=1E-2):
                print("Warning: Tracking field may not be relaible.")
                self._warning_not_printed = False

            # Calculate the tracking field using Eq. (2) of Phys. Rev. Lett. 118, 083201 (2017)
            self.E.append(
                -4. * (self.P_average[-1] - diff_Y) / self.dt + 2. * self.diff_V_average[-1] - self.E[-1]
            )

            # Take the laser field at mid point (to have a cubic accuracy propagator)
            self.laser_field = 0.5 * (self.E[-1] + self.E[-2])

            # # calculate the Ehrenfest theorems
            # self.get_Ehrenfest()
            #
            # self.E.append(
            #     self.diff_diff_Y[len(self.E) - 1] + self.diff_V_average[-1]
            # )
            # self.laser_field = 0.5 * (self.E[-1] + self.E[-2])

            # Calculate the exponent of the total potential energy
            ne.evaluate("exp(0.5j * dt * X * laser_field)", local_dict=self.__dict__, out=self.expVTotal)
            self.expVTotal *= self.expV

            # Propagate
            self.wavefunction *= self.expVTotal

            # going to the momentum representation
            self.wavefunction = fftpack.fft(self.wavefunction, overwrite_x=True)
            self.wavefunction *= self.expK

            # going back to the coordinate representation
            self.wavefunction = fftpack.ifft(self.wavefunction, overwrite_x=True)
            self.wavefunction *= self.expVTotal

            # normalize
            # this line is equivalent to
            # self.wavefunction /= np.sqrt(np.sum(np.abs(self.wavefunction)**2)*self.dX)
            self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dX)

            # increment time
            self.t += self.dt

        return self.wavefunction

    def propagate_TL(self, time_steps=1):
        """
        Propagate the wave function saved in self.wavefunction using the transform limited (TL) pulse
        :param time_steps: number of self.dt time increments to make
        :return: self.wavefunction
        """
        try:
            self.TL_laser_filed
        except AttributeError:
            raise AttributeError("Transform limited laser field (TL_laser_filed) has not been specified.")

        exp_TL_laser_field = "exp(0.5j * dt * X * %s)" % self.TL_laser_filed

        for _ in range(time_steps):
            # save laser field to calculate the Ehrenfest theorems
            self.E = [ne.evaluate(self.TL_laser_filed, local_dict=self.__dict__)]

            # verify the Ehrenfest theorems
            self.get_Ehrenfest()

            # increment half a time step
            self.t += 0.5 * self.dt

            # Calculate the exponent of the total potential energy
            ne.evaluate(exp_TL_laser_field, local_dict=self.__dict__, out=self.expVTotal)
            self.expVTotal *= self.expV

            # Propagate
            self.wavefunction *= self.expVTotal

            # going to the momentum representation
            self.wavefunction = fftpack.fft(self.wavefunction, overwrite_x=True)
            self.wavefunction *= self.expK

            # going back to the coordinate representation
            self.wavefunction = fftpack.ifft(self.wavefunction, overwrite_x=True)
            self.wavefunction *= self.expVTotal

            # normalize
            self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dX)

            # increment half a time step
            self.t += 0.5 * self.dt

        return self.wavefunction


    def get_Ehrenfest(self):
        """
        Calculate observables entering the Ehrenfest theorems
        """
        # evaluate the coordinate density
        np.abs(self.wavefunction, out=self.density)
        self.density *= self.density
        # normalize
        self.density /= self.density.sum()

        # save the current value of <X>
        self.X_average.append(
            ne.evaluate("sum(density * X)", local_dict=self.__dict__)
        )

        self.diff_V_average.append(
            ne.evaluate("sum(density * _diff_V)", local_dict=self.__dict__)
        )

        self.P_average_RHS.append(
            -self.diff_V_average[-1] + self.E[-1]
        )

        # save the potential energy
        self.hamiltonian_average.append(
            ne.evaluate("sum(density * _V)", local_dict=self.__dict__)
            # Add the contribution from laser field
            + self.E[-1] * self.X_average[-1]
        )

        # calculate density in the momentum representation
        ne.evaluate("(-1) ** k * wavefunction", local_dict=self.__dict__, out=self.wavefunction_p)
        self.wavefunction_p = fftpack.fft(self.wavefunction_p, overwrite_x=True)
        np.abs(self.wavefunction_p, out=self.density)
        self.density *= self.density
        # normalize
        self.density /= self.density.sum()

        # save the current value of <P>
        self.P_average.append(
            ne.evaluate("sum(density * P)", local_dict=self.__dict__)
        )

        # add the kinetic energy to get the hamiltonian
        self.hamiltonian_average[-1] += \
            ne.evaluate("sum(density * _K)", local_dict=self.__dict__)

    def set_wavefunction(self, wavefunc):
        """
        Set the initial wave function
        :param wavefunc: 1D numpy array or string containing the wave function
        :return: self
        """
        if isinstance(wavefunc, str):
            # wavefunction is supplied as a string
            ne.evaluate("%s + 0j" % wavefunc, local_dict=self.__dict__, out=self.wavefunction)

        elif isinstance(wavefunc, np.ndarray):
            # wavefunction is supplied as an array

            # perform the consistency checks
            assert wavefunc.shape == self.wavefunction.shape,\
                "The grid size does not match with the wave function"

            # make sure the wavefunction is stored as a complex array
            np.copyto(self.wavefunction, wavefunc.astype(np.complex))

        else:
            raise ValueError("wavefunc must be either string or numpy.array")

        # normalize
        self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dX)

        return self

    def set_groundstate(self, nsteps=1000):
        """
        Set the initial condition to be a ground state calculated via the imaginary time propagation
        :param nsteps: number of imaginary time steps to be taken
        :return: self
        """
        self.set_wavefunction("exp(-X ** 2)")

        # Pre-calculate the exponential of the potential without the laser field
        ImgTExpV = ne.evaluate("(-1) ** k * exp(-0.5 * dt * (%s))" % self.V, local_dict=self.__dict__)

        ImgTExpK = np.exp(-0.5 * self.dt * self.P ** 2)

        # Imaginary time propagation
        for _ in range(nsteps):
            self.wavefunction *= ImgTExpV

            # going to the momentum representation
            self.wavefunction = fftpack.fft(self.wavefunction, overwrite_x=True)
            self.wavefunction *= ImgTExpK

            # going back to the coordinate representation
            self.wavefunction = fftpack.ifft(self.wavefunction, overwrite_x=True)
            self.wavefunction *= ImgTExpV

            # normalize
            self.wavefunction /= linalg.norm(self.wavefunction) * np.sqrt(self.dX)

        return self

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    # Plotting facility
    import matplotlib.pyplot as plt

    # Use the documentation string for the developed class
    print(SDMSchrodinger1D.__doc__)

    for omega in [4., ]:

        t = np.linspace(-5, 5, 600)

        # save parameters as a separate bundle
        harmonic_osc_params = dict(
            X_gridDIM=512,
            X_amplitude=20.,
            dt=0.01,
            t=0.,

            omega=omega,

            V="0.5 * omega ** 2 * X ** 2",
            diff_V="2 * 0.5 * omega ** 2 * X",

            # Y = np.sin(0.1 * np.arange(600)) ** 2,
            Y=5 * np.random.rand() * np.sin(np.random.rand() * t) *  np.exp(-t ** 2),
        )

        ##################################################################################################

        # create the harmonic oscillator with time-independent hamiltonian
        harmonic_osc = SDMSchrodinger1D(**harmonic_osc_params)

        # set the initial condition
        harmonic_osc.set_wavefunction(
            "exp(-(X) ** 2)"
        )

        # propagate till time T and for each time step save a probability density
        wavefunctions = [harmonic_osc.propagate().copy() for _ in range(harmonic_osc.Y.size)]

        plt.title("Test 1: Time evolution of harmonic oscillator with $\\omega$ = %.2f (a.u.)" % omega)

        # plot the time dependent density
        from matplotlib.colors import LogNorm

        plt.imshow(
            np.abs(wavefunctions)**2,
            # some plotting parameters
            origin='lower',
            extent=[harmonic_osc.X.min(), harmonic_osc.X.max(), 0., harmonic_osc.t],
            norm=LogNorm(1e-7, 1.),
        )
        plt.xlabel('coordinate $x$ (a.u.)')
        plt.ylabel('time $t$ (a.u.)')
        plt.show()

        ##################################################################################################

        plt.subplot(131)
        plt.title("Verify the first Ehrenfest theorem")

        times = harmonic_osc.dt * np.arange(len(harmonic_osc.X_average))
        plt.plot(
            times,
            np.gradient(harmonic_osc.X_average, harmonic_osc.dt),
            '-r',
            label='$d\\langle\\hat{x}\\rangle / dt$'
        )
        plt.plot(times, harmonic_osc.X_average_RHS, '--b', label='$\\langle\\hat{p}\\rangle$')
        plt.legend()
        plt.ylabel('momentum')
        plt.xlabel('time $t$ (a.u.)')

        plt.subplot(132)
        plt.title("Verify the second Ehrenfest theorem")

        plt.plot(
            times,
            np.gradient(harmonic_osc.P_average, harmonic_osc.dt),
            '-r',
            label='$d\\langle\\hat{p}\\rangle / dt$'
        )
        plt.plot(times, harmonic_osc.P_average_RHS, '--b', label='$\\langle -U\'(\\hat{x})\\rangle$')
        plt.legend()
        plt.ylabel('force')
        plt.xlabel('time $t$ (a.u.)')

        plt.subplot(133)
        plt.title("The expectation value of the hamiltonian")

        # Analyze how well the energy was preserved
        h = np.array(harmonic_osc.hamiltonian_average)
        print(
            "\nHamiltonian is preserved within the accuracy of %.2e percent" % (100. * (1. - h.min() / h.max()))
        )

        plt.plot(times, h)
        plt.ylabel('energy')
        plt.xlabel('time $t$ (a.u.)')

        plt.show()

        ##################################################################################################

        def FT(A):
            """
            Fourier transform
            :param A:  1D numpy.array
            :return:
            """
            A = np.array(A)
            minus_one = (-1) ** np.arange(A.size)
            return minus_one * np.fft.fft(minus_one * A)

        plt.title("Laser field and response")

        plt.subplot(121)
        FT_Y = np.abs(FT(harmonic_osc.Y)) ** 2
        FT_Y /= FT_Y.max()
        plt.plot(FT_Y, label='target')

        FT_E = np.abs(FT(harmonic_osc.E)) ** 2
        FT_E /= FT_E.max()
        plt.plot(FT_E, label='laser field')

        plt.legend()
        plt.xlabel("$\omega$")
        plt.ylabel("$abs(FFT($\\cdot$))$")

        plt.subplot(122)
        plt.title("Target and objective")
        plt.plot(times, harmonic_osc.Y)
        plt.plot(times, harmonic_osc.X_average)
        plt.xlabel("time $(t)$")

        plt.show()