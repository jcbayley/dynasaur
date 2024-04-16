import numpy as np


def acceleration_1pn(m0, m1, r, v, rvect, vvect, G, c):

    prefact = 4*(G**2)/(5*(c**5)*(r**3))*m0*m1*(m1/(m0 + m1))

    fact1 = rvect*np.dot(rvect, vvect)*((34/3)*G*(m0+m1)/r + 6*v**2)

    fact2 = vvect*(-6*G*(m0+m1)/r - 2*v**2)
    #print(prefact, fact1, fact2)
    #print(c,G,m0,m1, r, v)
    return prefact * (fact1 + fact2)

def acceleration_c2(m1, m2, r12, v12, r1, r2, v1, v2, G, c):
    """taken from https://doi.org/10.12942/lrr-2014-2

    Args:
        m1: 
        m2: 
        r1 (_type_): _description_
        r2 (_type_): _description_
        v1 (_type_): _description_
        v2 (_type_): _description_
        G (_type_): _description_
        c (_type_): _description_

    Returns:
        _type_: _description_
    """
    #r12 = np.sum((r1 - r2)**2)
    n12 = (r1 - r2)/r12
    #v12 = v1 - v2

    fact1 = 5*(G**2)*m1*m2/(r12**3)

    fact2 = 4*(G**2)*(m2**2)/(r12**2)

    fact3 = G*m2/(r12**2)

    bracket1 = (3/2)*np.dot(n12, v2)**2 - np.dot(v1, v1) + 4*np.dot(v1,v2) - 2*np.dot(v2, v2)

    term1 = fact1 + fact2 + fact3*bracket1

    fact4 = G*m2/(r12**2)

    bracket2 = 2*np.dot(n12, v1) - 2*np.dot(n12,v2)

    term2 = fact4*bracket2

    return (1/c**2)*(term1*n12 + term2*v12)

def acceleration_c4(m1, m2, r12, v12, r1, r2, v1, v2, G, c):
    """taken from https://doi.org/10.12942/lrr-2014-2

    Args:
        r1 (_type_): _description_
        r2 (_type_): _description_
        v1 (_type_): _description_
        v2 (_type_): _description_
        G (_type_): _description_
        c (_type_): _description_

    Returns:
        _type_: _description_
    """
    #r12 = np.sum((r1 - r2)**2)
    n12 = (r1 - r2)/r12
    #v12 = v1 - v2
    term1 = (
        (-57 * G ** 3 * m1 ** 2 * m2 ** 2 / (4 * r12 ** 4)) +
        (-69 * G ** 3 * m1 * m2 ** 2 / (2 * r12 ** 4)) +
        (-9 * G ** 3 * m2 ** 2 / (r12 ** 4)) +
        (G * m2 / (r12 ** 2)) * (
            (-15 / 8) * (np.dot(n12,v2) ** 4) +
            (3 / 2) * (np.dot(n12, v2) ** 2 * np.dot(v1, v1)) -
            6 * (np.dot(n12, v2) ** 2 * np.dot(v1, v2)) -
            2 * np.dot(v1, v2) ** 2 +
            (9 / 2) * (np.dot(n12, v2) ** 2 * np.dot(v2, v2)) +
            4 * np.dot(v1, v2) * np.dot(v2, v2) -
            2 * (np.dot(v2, v2) ** 4)
        ) +
        (G ** 2 * m1 * m2 / (r12 ** 3)) * (
            (39 / 2) * (np.dot(n12, v1) ** 2) -
            39 * (np.dot(n12 , v1) * np.dot(n12, v2)) +
            (17 / 2) * (np.dot(n12, v2) ** 2) -
            (15 / 4) * np.dot(v1, v1) -
            (5 / 2) * np.dot(v1 , v2) +
            (5 / 4) * np.dot(v2, v2)
        ) +
        (G ** 2 * m2 ** 2 / (r12 ** 3)) * (
            2 * (np.dot(n12, v1) ** 2) -
            4 * (np.dot(n12, v1) * np.dot(n12 , v2)) -
            6 * (np.dot(n12, v2) ** 2) -
            8 * np.dot(v1, v2) +
            4 * np.dot(v2, v2)
        )
    )

    term2 = (
        (G ** 2 * m2 ** 2 / (r12 ** 3)) * (
            -2 * (np.dot(n12, v1)) -
            2 * np.dot(n12, v2)
        ) +
        (G ** 2 * m1 * m2 / (r12 ** 3)) * (
            (-63 / 4) * np.dot(n12, v1) +
            (55 / 4) * np.dot(n12, v2)
        ) +
        (G * m2 / (r12 ** 2)) * (
            -6 * (np.dot(n12, v1) * np.dot(n12, v2) ** 2) +
            (9 / 2) * (np.dot(n12, v2) ** 3) +
            (np.dot(n12, v2) * np.dot(v1, v1)) -
            4 * (np.dot(n12, v1) * np.dot(v2, v2)) +
            4 * (np.dot(n12, v2) * np.dot(v2, v2)) +
            4 * (np.dot(n12, v1) * np.dot(v2, v2)) -
            5 * (np.dot(n12, v2) * np.dot(v2, v2))
        )
    )

    return (1/c**4) * (term1 * n12 + term2 * v12)

def newton_derivative(
    t, 
    x_posvels, 
    masses, 
    n_dimensions=3, 
    EPS=1e-16,
    G=6.67e-11,
    c=3e8,
    correction_1pn=False,
    correction_2pn=False
    ):
    """compute the derivative of the position and velocity

    Args:
        x_positions (_type_): _description_
        masses (_type_): _description_
    """
    n_masses = len(masses)

    # define some constants 
    #G = 6.67e-11
    #c = 3e8
    # compute G in gaussian gravitational units to prevent overflows
    # G * (s/day)/(m/Au) * Msun(kg)

    #G_ggravunits = G * ((second_scale**2)/((distance_scale)**3)) * mass_scale
    #c_ggravunits = c * ((second_scale**2)/(distance_scale))

    # reshape posvels to more useful shape
    x_posvels = x_posvels.reshape(n_masses, 2*n_dimensions)
    x_positions = x_posvels[:, 0:n_dimensions]
    x_vels = x_posvels[:, n_dimensions:2*n_dimensions]

    seps = np.zeros((n_masses, n_masses))

    x_derivative = np.zeros((n_masses, 2*n_dimensions))
    for i, mass_1 in enumerate(masses):
        x_derivative[i][0:n_dimensions] = x_vels[i]
        for j, mass_2 in enumerate(masses):
            if i == j: continue
            separation = np.sqrt(np.sum((x_positions[i] - x_positions[j])**2) + EPS)
            separation_cubed = separation**3
            #seps[i,j] = separation_cubed
            diff = x_positions[j] - x_positions[i]
            veldiff = x_vels[i] - x_vels[j]
            absvel = np.sqrt(np.sum((veldiff)**2))
            total_acc = G*mass_2*diff/separation_cubed
            if correction_1pn:
                acc_c2 = acceleration_c2(mass_1, mass_2, separation, veldiff, x_positions[i], x_positions[j], x_vels[i], x_vels[j], G, c)
                #print("c2", acc_c2/total_acc)
                total_acc += acc_c2
                #total_acc += acceleration_1pn(mass_1, mass_2, separation, absvel, diff, veldiff, G, c)
            if correction_2pn:
                acc_c4 = acceleration_c4(mass_1, mass_2, separation, veldiff, x_positions[i], x_positions[j], x_vels[i], x_vels[j], G, c)
                #print("c4", acc_c4/total_acc)
                total_acc += acc_c4

            x_derivative[i][n_dimensions:2*n_dimensions] += total_acc
         
        """
        # get all other masses but this one
        other_positions = np.delete(x_positions, i)
        other_vels = np.delete(x_vels, i)
        # compute separations
        rs = np.sqrt(np.sum((other_positions - x_positions[j])**2))
        """

    return x_derivative.flatten()

def newton_derivative_vect(
    t, 
    x_posvels, 
    masses, 
    factor=1, 
    n_dimensions=3, 
    EPS=1e-11,
    second_scale=86400,
    mass_scale=1.989e30,
    distance_scale=1.4e11,
    G=6.67e-11,
    c=3e8):
    """compute the derivative of the position and velocity

    Args:
        x_positions (_type_): (Nmasses*6, )
        masses (_type_): _description_
    """
    n_masses = len(masses)

    # define some constants 
    # compute G in gaussian gravitational units to prevent overflows
    # G * (s/day)/(m/Au) * Msun(kg)
    G_ggravunits = G * ((second_scale**2)/((distance_scale)**3)) * mass_scale
    c_ggravunits = c * ((second_scale**2)/(distance_scale))

    x_posvels = x_posvels.reshape(n_masses, 2*n_dimensions)
    x_derivative = np.zeros((n_masses, 2*n_dimensions))

    # compute separations of each object and the absolute distance
    diff_xyz = x_posvels[:,0:3,None].T - x_posvels[:,0:3,None] # Nx3xxN matrix
    #inv_r_cubed = (np.sum(diff_xyz**2, axis=1) + EPS)**(-1.5) # NxN matrix
    inv_r_cubed = (np.einsum("ijk,ijk->ik", diff_xyz, diff_xyz) + EPS)**-1.5
    #print("invdiff", np.sum(diff_xyz**2, axis=1) - inv_r_cubed2)

    # below two lines are the same
    # take matrix multiplication of masses with last dimensions
    #acceleration = G*(dxyz * inv_r_cubed) @ masses
    #acceleration = np.einsum("ijk,k", G*(dxyz * inv_r_cubed), masses)

    x_derivative[:, 0:3] = x_posvels[:, 3:6]
    #div_r_cubed = np.einsum("ijk, ik->ijk", G_ggravunits*diff_xyz, inv_r_cubed)
    x_derivative[:, 3:6] = np.einsum("ijk, ik, k->ij", G_ggravunits*diff_xyz, inv_r_cubed, masses)
    #x_derivative[:, 3:6] = np.einsum("ijk,k->ij", div_r_cubed, masses)

    return x_derivative.flatten()


