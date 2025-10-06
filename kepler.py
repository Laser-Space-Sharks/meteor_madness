from dataclasses import dataclass

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# constants in KKS
G = 6.674e-20
Me = 5.972e24
Re = 6378
Msol = 1.9885e30


@dataclass
class CartesianElement:
    r: np.ndarray[3]
    v: np.ndarray[3]
    mu:  np.float64      # standard gravitational parameter (GM_e or GM_sol)


@dataclass
class KeplerElement:
    sma:    np.float64   # semi-major axis
    ecc:    np.float64   # eccentricity
    argp:   np.float64   # argument of periapsis
    lasc:   np.float64   # longitude of the ascending node
    inc:    np.float64   # inclination
    nu:     np.float64   # true anomaly
    mu:     np.float64   # standard gravitational parameter


KEPLER_DEFAULT = (
    -12448.799216310952,
    1.5051654574733837,
    4.380587051837244,
    1.1390200269029194,
    2.4252340740654765,
    0.21630341652911592,
    G*Me
)


def cart2kepler(c: CartesianElement) -> KeplerElement:
    rmag = np.linalg.norm(c.r)

    h = np.cross(c.r, c.v)      # angular momentum
    hmag = np.linalg.norm(h)

    n = np.array([-h[1], h[0], 0])  # zhat x h, ascending node vector

    vsqr = np.dot(c.v, c.v)
    nmag = np.linalg.norm(n)

    if n[1] >= 0:
        lasc = math.acos(n[0]/nmag)
    else:
        lasc = 2*math.pi - math.acos(n[0]/nmag)

    evec = np.cross(c.v, h)/c.mu - c.r/rmag
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
    if abs(inc - 0) < EPSILON:
        lasc = 0
        if abs(ecc - 0) < EPSILON:
            argp = 0
        else:
            argp = math.acos(evec[0]/ecc)
    else:
        lasc = math.acos(n[0]/nmag)
        if n[1] < 0:
            lasc = 2*math.pi - lasc

        argp = math.acos(np.dot(evec, n)/(ecc*nmag))

    if abs(ecc - 0) < EPSILON:
        if abs(inc - 0) < EPSILON:
            nu = math.acos(c.r[0]/rmag)
            if c.v[0] > 0:
                nu = 2*math.pi - nu
        else:
            nu = math.acos(np.dot(c.r, n)/(rmag*nmag))
            if np.dot(c.v, n) > 0:
                nu = 2*math.pi - nu
    else:
        if evec[2] < 0:
            argp = 2*math.pi - argp

        nu = math.acos(np.dot(c.r, evec)/(rmag*ecc))
        if np.dot(c.r, c.v) < 0:
            nu = 2*math.pi - nu

    return KeplerElement(sma, ecc, argp, lasc, inc, nu, c.mu)


def kepler2cart(k: KeplerElement):
    rot = Rotation.from_euler("ZXZ", [-k.argp, -k.inc, -k.lasc]).as_matrix()
    coeff = k.sma*(1-k.ecc**2)
    rmag = coeff/(1+k.ecc*math.cos(k.nu))
    r = rmag*np.array([math.cos(k.nu), math.sin(k.nu), 0])

    v = math.sqrt(k.mu/coeff)*np.array([-math.sin(k.nu),
                                        k.ecc+math.cos(k.nu), 0])
    return CartesianElement(r @ rot, v @ rot, k.mu)


def conic_from_impact(lat: np.float32, lon: np.float32, v: np.ndarray, mu, samples):
    latr = lat * math.pi/180
    theta = math.pi/2 - latr
    lonr = lon * math.pi/180
    r = Re*np.array([math.sin(theta)*math.sin(lonr),
                     math.sin(theta)*math.cos(lonr), math.cos(theta)])
    kepler = cart2kepler(CartesianElement(r, v, mu))
    if kepler.ecc < 1:
        min_anomaly = -2*math.pi
    else:
        min_anomaly = -15/16*math.acos(-1/kepler.ecc)

    anomalies = np.linspace(min_anomaly, kepler.nu, samples)
    xs = np.empty(samples)
    ys = np.empty(samples)
    zs = np.empty(samples)
    new_kep = kepler
    for i in range(samples):
        new_kep = kepler
        new_kep.nu = anomalies[i]
        point = kepler2cart(new_kep)
        xs[i] = point.r[0]
        ys[i] = point.r[1]
        zs[i] = point.r[2]

    return (xs, ys, zs, np.linalg.norm(kepler2cart(kepler).v))


def kepler_conic(k: KeplerElement, samples):
    if k.ecc < 1:
        max_anomaly = math.pi
    else:
        max_anomaly = -15/16*math.acos(-1/k.ecc)
    anomalies = np.linspace(-max_anomaly, max_anomaly, samples)
    xs = np.empty(samples)
    ys = np.empty(samples)
    zs = np.empty(samples)
    i = 0
    while i < samples:
        new_k = k
        new_k.nu = anomalies[i]
        point = kepler2cart(new_k)
        if np.dot(point.r, point.r) < Re*Re:
            anomalies = np.linspace(-max_anomaly, anomalies[i-1], samples)
            i = 0
            continue
        xs[i] = point.r[0]
        ys[i] = point.r[1]
        zs[i] = point.r[2]
        i += 1

    return (xs, ys, zs)


def latlon_from_cartesian(r: np.ndarray):
    rhat = r/np.linalg.norm(r)
    lat = 90 - 180/math.pi * math.acos(rhat[2])
    rflat = r
    rflat[2] = 0
    rflathat = rflat/np.linalg.norm(rflat)
    lon = 180/math.pi * math.atan2(rflathat[1], rflathat[0])
    return (lat, lon)


def main() -> None:
    impact_conic = conic_from_impact(-40.7128, -74.0060, np.array([3, 15, 10]),
                                     G*Me, 500)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make sphere (Earth)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    ex = 6378 * np.outer(np.cos(u), np.sin(v))
    ey = 6378 * np.outer(np.sin(u), np.sin(v))
    ez = 6378 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(ex, ey, ez)

    ax.plot(impact_conic[0], impact_conic[1], impact_conic[2])
    ax.plot(impact_conic[0][-1], impact_conic[1][-1], impact_conic[2][-1], c="r", marker='o')

    ax.set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    main()
