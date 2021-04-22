import numpy as np
from scipy import interpolate

def checkDims(exp):

    if exp.ndim < 2:
        exp = exp

    else:
        xmean = np.mean(exp[:, 0])
        ymean = np.mean(exp[:, 1])
        zmean = np.mean(exp[:, 2])

        exp = np.array([xmean, ymean, zmean])

    return exp

def stationaryCompare(exp):

    """
    Compares the positions returned by PEPT algorithms from a single particle tracking test. A single stationary
    tracer is placed in the field-of-view. Exp is a (N, 3) np.array of x, y, z, positions. The position of the real
    position for comparison may be left as is, edited to a new positions, or changed to read in a new position from a
    file.
    """

    #real = np.loadtxt('Keys/Key_Stationary.csv')

    # Real position of the tracer
    x_real = 0
    y_real = 10
    z_real = 0

    # If multiple positions are given, compute the mean
    exp = checkDims(exp)

    # Break exp in to individual positional components
    x_exp = exp[0]
    y_exp = exp[1]
    z_exp = exp[2]

    # Compute individual error components
    x_error = abs(x_real - x_exp)
    y_error = abs(y_real - y_exp)
    z_error = abs(z_real - z_exp)

    # Compute 3D error
    error = np.sqrt(x_error ** 2 + y_error ** 2 + z_error ** 2)

    return error


def scatterCompare(exp):


    """
    Compares the positions returned by PEPT algorithms from a single particle tracking test. A single stationary
    tracer is placed in the field-of-view and placed inside a sphere of material to induce scattering. Exp is a (N, 3)
    np.array of x, y, z, positions. The position of the real position for comparison may be left as is, edited to a new
    positions, or changed to read in a new position from a file.
    """
    #real = np.loadtxt('Keys/Key_Scatter.csv')

    # Real particle position
    x_real = 0
    y_real = 20
    z_real = 0

    # If multiple positions are given, compute the mean
    exp = checkDims(exp)

    # Break exp in to individual positional components
    x_exp = exp[0]
    y_exp = exp[1]
    z_exp = exp[2]

    # Compute individual error components
    x_error = abs(x_real - x_exp)
    y_error = abs(y_real - y_exp)
    z_error = abs(z_real - z_exp)

    # Compute 3D error
    error = np.sqrt(x_error ** 2 + y_error ** 2 + z_error ** 2)

    return error


def fieldOfViewCompare(exp):

    """
    Compares the positions returned by PEPT algorithms from a single particle tracking test. A single tracer enters one
    side of the field-of-view then exits out the other. Exp is a (N, 4) np.array of t, x, y, z, positions. The position
    of the real tracer is described by a equation as a function of time. The position to compare is computed for the
    timestep associated with the PEPT detection.
    """

    # Read in PEPT detected trajectory
    t_exp = exp[:, 0]
    x_exp = exp[:, 1]
    y_exp = exp[:, 2]
    z_exp = exp[:, 3]

    # A tracer moves in a straight line at a velocity of 250 mm/s in the z direction
    v = 250  # mm/s
    x_real = 0
    y_real = 0
    z_real = -250 + (t_exp / 1000) * v

    # Computer individual error components
    x_error = abs(x_real - x_exp)
    y_error = abs(y_real - y_exp)
    z_error = abs(z_real - z_exp)

    # Compute instantaneous errors
    errors = np.sqrt(x_error**2+y_error**2+z_error**2)

    # Compute mean error
    error = np.mean(errors)

    # Compute standard deviation of the individual error components
    x_std = np.std(x_error)
    y_std = np.std(y_error)
    z_std = np.std(z_error)

    # Computer overall standard deviation of the 3D particle position error
    std = np.mean(np.sqrt(x_std ** 2 + y_std ** 2 + z_std ** 2))

    return error, std, errors


def velocityCompare(exp, real):

    """
    Compares the positions returned by PEPT algorithms from a single particle tracking test. A single tracer moves
    between two positions at constant velocity. Exp is a (N, 4) np.array of t, x, y, z, positions. The position of the
    real tracer is read in from a file and passed to this function for each of the different tests. The file contains a
    series of timesteps and positions for the prescribed movements. To compare to the associated PEPT detections, the
    PEPT detected timestep is used as the basis for linearly interpolating to the expected real position.
    """

    # Break up exp into time and position components
    t_exp = exp[:, 0]
    x_exp = exp[:, 1]
    y_exp = exp[:, 2]
    z_exp = exp[:, 3]

    # Break of real into individual time an position components
    t_real = real[:, 0]
    x_real = real[:, 1]
    y_real = real[:, 2]
    z_real = real[:, 3]

    # Create linear interpolation functions
    fx = interpolate.interp1d(t_real, x_real)
    fy = interpolate.interp1d(t_real, y_real)
    fz = interpolate.interp1d(t_real, z_real)

    # Use the PEPT detected timesteps to calculate the real particle position
    x_real = fx(t_exp)
    y_real = fy(t_exp)
    z_real = fz(t_exp)

    # Compute instantaneous error for the individual position components
    x_error = abs(x_real - x_exp)
    y_error = abs(y_real - y_exp)
    z_error = abs(z_real - z_exp)

    # Compute the mean 3D error
    error = np.mean(np.sqrt(x_error ** 2 + y_error ** 2 + z_error ** 2))

    # Compute the 1D standard deviation
    x_std = np.std(x_error)
    y_std = np.std(y_error)
    z_std = np.std(z_error)

    # Compute the 3D standard deviation
    std = np.mean(np.sqrt(x_std ** 2 + y_std ** 2 + z_std ** 2))

    return error, std


def seperationCompare(exp1, exp2, real):

    """
    Compares the positions returned by PEPT algorithms from a two particle tracking test. Two tracers are separated by
    small distances, testing the ability to differentiate increasing small tracer separations. Exp is a (N, 3) np.array
    x, y, z, positions. The position of the real tracer is passed to this function for each of the different tests. The
    centre-to-centre separations used in this test are 2, 3, 4, 6, 8, 10, 12, 16, 20, and 30 mm. Tests are conducted in
    both the transaxial (x) and axial (z) orientations of the detector, with the other position components left as 0 mm.
    """

    # Particle 1 position componets
    x1_exp = exp1[:, 0]
    y1_exp = exp1[:, 1]
    z1_exp = exp1[:, 2]

    # Particle 2 position componets
    x2_exp = exp2[:, 0]
    y2_exp = exp2[:, 1]
    z2_exp = exp2[:, 2]

    # Real particle 1 position components
    x1_real = -real[0]
    y1_real = -real[1]
    z1_real = -real[2]

    # Real particle 2 position components
    x2_real = real[0]
    y2_real = real[1]
    z2_real = real[2]

    # Compute the instantaneous 1D error for particle 1
    x1_error = abs(x1_real - x1_exp)
    y1_error = abs(y1_real - y1_exp)
    z1_error = abs(z1_real - z1_exp)

    # Compute mean 3D error for particle 1
    error1 = np.mean(np.sqrt(x1_error ** 2 + y1_error ** 2 + z1_error ** 2))

    # Compute 1D standard deviation for particle 1 positions
    x1_std = np.std(x1_error)
    y1_std = np.std(y1_error)
    z1_std = np.std(z1_error)

    # Compute 3D standard deviation for particle 1
    std1 = np.sqrt(x1_std ** 2 + y1_std ** 2 + z1_std ** 2)

    # Compute the instantaneous 1D error for particle 2
    x2_error = abs(x2_real - x2_exp)
    y2_error = abs(y2_real - y2_exp)
    z2_error = abs(z2_real - z2_exp)

    # Compute mean 3D error for particle 2
    error2 = np.mean(np.sqrt(x2_error ** 2 + y2_error ** 2 + z2_error ** 2))

    # Compute 1D standard deviation for particle 2 positions
    x2_std = np.std(x2_error)
    y2_std = np.std(y2_error)
    z2_std = np.std(z2_error)

    # Compute 3D standard deviation for particle 2
    std2 = np.mean(np.sqrt(x2_std ** 2 + y2_std ** 2 + z2_std ** 2))

    # Mean error for both particles
    error = (error1 + error2) / 2

    # Mean standard deviation for both particles
    std = (std1 + std2) / 2

    return error, std

def linkingCompare(exp, real):

    """
    Compares the positions returned by PEPT algorithms from a three particle tracking test. Three tracers are separated
    by a constant separation and continuously tracked as they move about the surface of the sphere, testing the ability
    link positions into trajectories for multiple particles simultaneously. Exp is a (N, 4) np.array t, x, y, z,
    positions. The position of the real tracer is passed to this function for each of the different tests. The PEPT
    trajectories must be correctly associated with the right real trajectory before passing to this function.
    """

    # Break up exp into individual time and position components
    t_exp = exp[:, 0]
    x_exp = exp[:, 1]
    y_exp = exp[:, 2]
    z_exp = exp[:, 3]

    # Break up the real trajectory into individual time and position components
    t_real = real[:, 0]
    x_real = real[:, 1]
    y_real = real[:, 2]
    z_real = real[:, 3]

    # Create interpolation functions for the real trajectory
    fx = interpolate.interp1d(t_real, x_real)
    fy = interpolate.interp1d(t_real, y_real)
    fz = interpolate.interp1d(t_real, z_real)

    # Calculate the expected real tracer position for each detected time
    x_real = fx(t_exp)
    y_real = fy(t_exp)
    z_real = fz(t_exp)

    # Compute the instantaneous 1D errors
    x_error = abs(x_real - x_exp)
    y_error = abs(y_real - y_exp)
    z_error = abs(z_real - z_exp)

    # Compute the mean 3D error
    error = np.mean(np.sqrt(x_error ** 2 + y_error ** 2 + z_error ** 2))

    # Compute the 1D standard deviation
    x_std = np.std(x_error)
    y_std = np.std(y_error)
    z_std = np.std(z_error)

    # Compute the 3D standard deviation
    std = np.sqrt(x_std ** 2 + y_std ** 2 + z_std ** 2)

    # Record how many timesteps are given for the PEPT detected trajectories
    time = len(t_exp)

    return error, std, time


def multipleCompare(exp, real):

    """
    Compares the positions returned by PEPT algorithms from a multiple particle tracking test. A random number of
    static tracers are placed with the detection volume of the detectors. An array of the PEPT detected positions for
    each test are compared to the real position. A tracer is considered found if it is the nearest tracer to the PEPT
    returned positions. A list of the found tracers is scanned for unique IDs and returned as the number of particles
    found. Additionally, the mean 3D error and the 3D standard deviation are returned. This is meant to testing the
    ability to track high numbers of particles simultaneously. Exp is a (N, 3) np.array x, y, z, positions. The position
    of the real tracers is passed to this function for each of the different tests.
    """

    # Break exp into position components
    x_exp = exp[:, 0]
    y_exp = exp[:, 1]
    z_exp = exp[:, 2]

    # Break up real into position components
    x_real = real[:, 0]
    y_real = real[:, 1]
    z_real = real[:, 2]

    # Create mean 3D error array
    errors_mean = []
    # Allocate space for instantaneous 3D errors
    errors = np.zeros(len(x_exp))
    # Create a 3D standard deviation array
    stds = []
    # Allocate space for the nearest particle IDs
    IDs = np.zeros(len(x_exp))

    # Loop over all detected positions
    for j in range(len(x_exp)):

        # Allocate space for instananeous errors over all real particles
        errors_ind = np.zeros(len(x_real))

        # Loop over all real particles
        for k in range(len(x_real)):

            # Compute 1D error between exp and all real particles
            x_error = abs(x_exp[j] - x_real[k])
            y_error = abs(y_exp[j] - y_real[k])
            z_error = abs(z_exp[j] - z_real[k])

            # Compute 3D error between exp and all real particles
            error = np.sqrt(x_error ** 2 + y_error ** 2 + z_error ** 2)

            # Fill in instanteous error array
            errors_ind[k] = error

        # Find the closest real particle and record the error
        IDs[j] = np.argmin(errors_ind)
        errors[j] = np.min(errors_ind)

    # The length of unique particle IDs is the number of particles found
    N_unique = len(np.unique(IDs))

    # Record mean paritcle 3D error
    errors_mean.append(np.mean(errors))

    # Record the 3D standard deviation
    stds.append(np.std(errors))

    return errors_mean, stds, N_unique


def closepackedCompare(exp, real):

    """
    Compares the positions returned by PEPT algorithms from a multiple particle tracking test. Arrangements of static
    racers are placed with the detection volume of the detectors. An array of the PEPT detected positions for
    each test are compared to the real position. A tracer is considered found if it is the nearest tracer to the PEPT
    returned positions. A list of the found tracers is scanned for unique IDs and returned as the number of particles
    found. Additionally, the mean 3D error and the 3D standard deviation are returned. This is meant to testing the
    ability to track high numbers of particles in close proximity to each other with high degrees of symmetry
    simultaneously. Exp is a (N, 3) np.array x, y, z, positions. The position of the real tracers is passed to this
    function for each of the different tests.
    """

    # Break exp into components
    x_exp = exp[:, 0]
    y_exp = exp[:, 1]
    z_exp = exp[:, 2]

    # Break real into components
    x_real = real[:, 0]
    y_real = real[:, 1]
    z_real = real[:, 2]

    # Create lists for the overall mean 3D errors and 3D standard deviations for each test
    errors = []
    stds = []

    # Loop over all real positions
    for k in range(len(x_real)):

        # Compute the 1D error compared to each real position
        x_error = abs(x_exp - x_real[k])
        y_error = abs(y_exp - y_real[k])
        z_error = abs(z_exp - z_real[k])

        # Compute the 3D error
        error = np.mean(np.sqrt(x_error ** 2 + y_error ** 2 + z_error ** 2))

        # Compute the 1D standard deviations
        x_std = np.std(x_error)
        y_std = np.std(y_error)
        z_std = np.std(z_error)

        # Compute the 3D standard deviation
        std = np.mean(np.sqrt(x_std ** 2 + y_std ** 2 + z_std ** 2))

        # Append to previous arrays
        errors.append(error)
        stds.append(std)

    # Find the closest real particle
    error = np.min(errors)
    std = stds[np.where(errors==np.min(errors))[0][0]]

    return error, std


