from dataclasses import dataclass

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# constants in MKS
G = 6.674e-20
Me = 5.972e24
Msol = 1.9885e30


@dataclass
class CartesianElement:
    r: np.ndarray[3]
    v: np.ndarray[3]
    mu:  np.float64      # standard gravitational parameter (GM_e or GM_sol)


@dataclass
class KeplerElement:
    sma:    np.float64   # semi-major axis
    ecc:      np.float64   # eccentricity
    argp:   np.float64   # argument of periapsis
    lasc:   np.float64   # longitude of the ascending node
    inc:    np.float64   # inclination
    nu:     np.float64   # true anomaly
    mu:     np.float64   # standard gravitational parameter

# Earth J2000


def cart2kepler(c: CartesianElement) -> KeplerElement:
    rmag = np.linalg.norm(c.r)

    h = np.cross(c.r, c.v)      # angular momentum
    hmag = np.linalg.norm(h)

    n = np.array([-h[1], h[0], 0])  # zhat x h, ascending node vector
    nmag = np.linalg.norm(n)

    vsqr = np.dot(c.v, c.v)

    evec = 1/c.mu*((vsqr - c.mu/rmag)*c.r - np.dot(c.r, c.v)*c.v)
    ecc = np.linalg.norm(evec)

    E = vsqr/2 - c.mu/rmag
    if ecc != 1:
        sma = -c.mu/(2*E)
    else:
        sma = np.inf

    EPSILON = 1e-15

    # inclination is the angle between h and its z component
    inc = math.acos(h[2]/hmag)

    # if the inclination is small, the longitude of the ascending node is 0.
    # same goes for the argument of the periapsis with small eccentricity
    if abs(inc) < EPSILON:
        lasc = 0
        if abs(ecc) < EPSILON:
            argp = 0
        else:
            argp = math.acos(evec[0] / ecc)
    else:
        lasc = math.acos(n[0]/nmag)
        if n[1] < 0:
            lasc = 2*math.pi - lasc

        argp = math.acos(np.dot(evec, n)/(ecc*nmag))

    if abs(ecc) < EPSILON:
        if abs(inc) < EPSILON:
            nu = math.acos(c.r[0]/rmag)
            if c.v[0] > 0:
                nu = 2*math.pi - nu
        else:
            nu = math.acos(np.dot(c.r, n) / (rmag * nmag))
            if np.dot(c.v, n) > 0:
                nu = 2*math.pi - nu
    else:
        if evec[2] < 0:
            argp = 2*math.pi - argp

        nu = math.acos(np.dot(c.r, evec)/(ecc*rmag))
        if np.dot(c.r, c.v) < 0:
            nu = 2*math.pi - nu

    return KeplerElement(sma, ecc, argp, lasc, inc, nu, c.mu)


def kepler2cart(k: KeplerElement):
    if k.ecc < 1:  # elliptical case
        cosE = (k.ecc+math.cos(k.nu))/(1+k.ecc*math.cos(k.nu))
        rmag = k.sma*(1-k.ecc*cosE)
        r = rmag*np.array([math.cos(k.nu), math.sin(k.nu), 0])
        v = math.sqrt(k.mu*k.sma)/rmag*np.array([-math.sin(math.acos(cosE)),
                                                 math.sqrt(1-k.ecc**2)*cosE, 0])
        rot = Rotation.from_euler("ZXZ", [-k.argp, -k.inc, -k.lasc])
        return CartesianElement(rot.as_matrix() @ r, rot.as_matrix() @ v, k.mu)
    if k.ecc == 1:  # parabolic case
        return k.ecc

    # hyperbolic case
    rmag = k.sma*(1-k.ecc**2)/(1+k.ecc*math.cos(k.nu))
    r = rmag*np.array([math.cos(k.nu), math.sin(k.nu), 0])
    rot = Rotation.from_euler("ZXZ", [-k.argp, -k.inc, -k.lasc])
    return CartesianElement(rot.as_matrix() @ r, 0, k.mu)


def main() -> None:
    r_init = np.array([7548, 0, 0])
    v_init = np.array([0, 9, 5])
    cart_init = CartesianElement(r_init, v_init, G*Me)

    # Solve for the Kepler orbit from r_init and v_init
    kepler = cart2kepler(cart_init)
    print(kepler.sma)
    print(kepler.ecc)
    print(kepler.inc)
    print(kepler.argp)
    print(kepler.lasc)

    theta_range = [0, 2*math.pi]
    if kepler.ecc > 1:
        theta_range = 15/16*np.array([-math.acos(-1/kepler.ecc),
                                      math.acos(-1/kepler.ecc)])

    thetas = np.linspace(theta_range[0], theta_range[1], 1000)
    xs = np.empty((thetas.shape[0]))
    ys = np.empty((thetas.shape[0]))
    zs = np.empty((thetas.shape[0]))
    for i in range(len(thetas)):
        new_kep = kepler
        new_kep.nu = thetas[i]
        cart = kepler2cart(new_kep).r
        xs[i] = cart[0]
        ys[i] = cart[1]
        zs[i] = cart[2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make sphere (Earth)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    ex = 6348 * np.outer(np.cos(u), np.sin(v))
    ey = 6348 * np.outer(np.sin(u), np.sin(v))
    ez = 6348 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(ex, ey, ez)

    ax.plot(xs, ys, zs)

    ax.set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    main()
