from dataclasses import dataclass

import numpy as np
import math
import matplotlib.pyplot as plt

# constants in MKS
G = 6.674e-11
Me = 5.972e24
Msol = 1.9885e30


@dataclass
class CartesianElement:
    pos: np.ndarray[3]
    vel: np.ndarray[3]
    mu:  np.float64      # standard gravitational parameter (GM_e or GM_sol)


@dataclass
class KeplerElement:
    smaxis: np.float64   # semi-major axis
    ecc:    np.float64   # eccentricity
    argp:   np.float64   # argument of periapsis
    lasc:   np.float64   # longitude of the ascending node
    inc:    np.float64   # inclination
    nu:     np.float64   # true anomaly
    mu:     np.float64   # standard gravitational parameter


def rotx(theta):
    return np.array([[1, 0, 0],
                     [0, math.cos(theta), -math.sin(theta)],
                     [0, math.sin(theta), math.cos(theta)]])


def roty(theta):
    return np.array([[math.cos(theta), 0, math.sin(theta)],
                     [0, 1, 0],
                     [-math.sin(theta), 0, math.cos(theta)]])


def rotz(theta):
    return np.array([[math.cos(theta), -math.sin(theta), 0],
                     [math.sin(theta), math.cos(theta), 0],
                     [0, 0, 1]])


def cart2kepler(c: CartesianElement) -> KeplerElement:
    h = np.cross(c.pos, c.vel)      # angular momentum
    hmag = np.linalg.norm(h)

    n = np.array([-h[1], h[0], 0])  # zhat x h, node vector
    nmag = np.linalg.norm(n)

    rmag = np.linalg.norm(c.pos)
    vsqr = np.dot(c.vel, c.vel)

    ev = 1/c.mu*((vsqr - c.mu/rmag)*c.pos - np.dot(c.pos, c.vel)*c.vel)
    e = np.linalg.norm(ev)

    E = vsqr/2 - c.mu/rmag
    if e != 1:
        a = -c.mu/(2*E)
    else:
        a = np.inf

    EPSILON = 1e-15

    # inclination is the angle between h and its z component
    inc = math.acos(h[2]/hmag)

    # if the inclination is small, the longitude of the ascending node is 0.
    # same goes for the argument of the periapsis with small eccentricity
    if abs(inc) < EPSILON:
        lasc = 0
        if abs(e) < EPSILON:
            argp = 0
        else:
            argp = math.acos(ev[0] / e)
    else:
        lasc = math.acos(n[0]/nmag)
        if n[1] < 0:
            lasc = 2*math.pi - lasc

        argp = math.acos(np.dot(ev, n)/(e*nmag))

    if abs(e) < EPSILON:
        if abs(inc) < EPSILON:
            nu = math.acos(c.pos[0]/rmag)
            if c.vel[0] > 0:
                nu = 2*math.pi - nu
        else:
            nu = math.acos(np.dot(c.pos, n) / (rmag * nmag))
            if np.dot(c.vel, n) > 0:
                nu = 2*math.pi - nu
    else:
        if ev[2] < 0:
            argp = 2*math.pi - argp

        nu = math.acos(np.dot(c.pos, ev)/(e * rmag))
        if np.dot(c.pos, c.vel) < 0:
            nu = 2*math.pi - nu

    return KeplerElement(a, e, argp, lasc, inc, nu, c.mu)


def kepler2cart(k: KeplerElement):
    if k.ecc < 1:  # elliptical case
        cosE = (k.ecc+math.cos(k.nu))/(1+k.ecc*math.cos(k.nu))
        rmag = k.smaxis*(1-k.ecc*cosE)
        r = rmag*np.array([math.cos(k.nu), math.sin(k.nu), 0])
        v = (math.sqrt(k.mu*k.smaxis)/rmag)*np.array([-math.sin(math.acos(cosE)),
                                                      math.sqrt(1-k.ecc**2)*cosE, 0])
        print(v)
        rot = rotz(-k.argp) @ rotx(-k.inc) @ rotz(-k.lasc)
        return CartesianElement(rot @ r, rot @ v, k.mu)
    if k.ecc == 1:  # parabolic case
        return k.ecc

    # hyperbolic case
    rmag = k.smaxis*(1-k.ecc**2)/(1+k.ecc*math.cos(k.nu))
    r = rmag*np.array([math.cos(k.nu), math.sin(k.nu), 0])
    rot = rotz(-k.argp) @ rotx(-k.inc) @ rotz(-k.lasc)
    return CartesianElement(rot @ r, 0, k.mu)


def main() -> None:
    r_init = np.array([7548e3, 0, 0])
    v_init = np.array([0, 9e3, 5e3])
    cart_init = CartesianElement(r_init, v_init, G*Me)

    # Solve for the Kepler orbit from r_init and v_init
    kepler = cart2kepler(cart_init)
    print(kepler.smaxis)
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
        cart = kepler2cart(new_kep).pos
        xs[i] = cart[0]
        ys[i] = cart[1]
        zs[i] = cart[2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make sphere (Earth)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    ex = 6348e3 * np.outer(np.cos(u), np.sin(v))
    ey = 6348e3 * np.outer(np.sin(u), np.sin(v))
    ez = 6348e3 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(ex, ey, ez)

    ax.plot(xs, ys, zs)

    ax.set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    main()
