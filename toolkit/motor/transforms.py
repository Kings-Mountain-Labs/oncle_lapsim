import numpy as np
import cmath
import math

def inverse_clarke_transform_matrix(alpha, beta, zero):
    """
    Perform the inverse Clarke transform using matrix operations.

    Args:
    alpha (float): Alpha quantity from Clarke transform
    beta (float): Beta quantity from Clarke transform
    zero (float): Zero sequence component

    Returns:
    np.array: [a, b, c] quantities
    """
    T_inv_clarke = np.array(
        [
            [1, 0, 1 / np.sqrt(3)],
            [-0.5, np.sqrt(3) / 2, 1 / np.sqrt(3)],
            [-0.5, -np.sqrt(3) / 2, 1 / np.sqrt(3)],
        ]
    )

    alpha_beta_zero = np.array([alpha, beta, zero])
    abc = np.dot(T_inv_clarke, alpha_beta_zero)
    return abc


def inverse_park_transform_matrix(d, q, theta):
    """
    Perform the inverse Park transform using matrix operations.

    Args:
    d (float): d-axis quantity from Park transform
    q (float): q-axis quantity from Park transform
    theta (float): Rotor angle in radians

    Returns:
    np.array: [alpha, beta] quantities
    """
    T_inv_park = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    dq = np.array([d, q])
    alpha_beta = np.dot(T_inv_park, dq)
    return alpha_beta


def calculate_real_power(V_a, V_b, V_c, I_a, I_b, I_c):
    """
    Calculate the total real power in an unbalanced sinusoidal three-phase system
    using symmetrical components theory with positive sequence phasors.

    Parameters:
    V_a, V_b, V_c : complex
        Phasor voltages of phases a, b, and c.
    I_a, I_b, I_c : complex
        Phasor currents of phases a, b, and c.

    Returns:
    float
        Total real power.
    """
    # Define the phase shift operator a
    a = cmath.exp(1j * 2 * cmath.pi / 3)

    # Calculate the positive sequence components of voltages
    V_0 = (V_a + V_b + V_c) / 3
    V_1 = (V_a + a * V_b + a**2 * V_c) / 3
    V_2 = (V_a + a**2 * V_b + a * V_c) / 3

    # Calculate the positive sequence components of currents
    I_0 = (I_a + I_b + I_c) / 3
    I_1 = (I_a + a * I_b + a**2 * I_c) / 3
    I_2 = (I_a + a**2 * I_b + a * I_c) / 3

    # Calculate the magnitudes of V_1 and I_1
    V_1_magnitude = abs(V_1)
    I_1_magnitude = abs(I_1)

    # Calculate the phase angles of V_1 and I_1
    V_1_angle = cmath.phase(V_1)
    I_1_angle = cmath.phase(I_1)

    # Calculate the total real power
    P_total = 3 * V_1_magnitude * I_1_magnitude * math.cos(V_1_angle - I_1_angle)
    S = np.sqrt((V_a**2 + V_b**2 + V_c**2) * (I_a**2 + I_b**2 + I_c**2))
    pf = P_total / S

    return P_total, pf


def calc_abc(t, d, q):
    alpha_beta = inverse_park_transform_matrix(d, q, t)
    return inverse_clarke_transform_matrix(alpha_beta[0], alpha_beta[1], 0)