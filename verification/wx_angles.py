# !/usr/bin/env python3
# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from math import pi
from typing import List, Tuple

import numpy as np
from numpy.polynomial.polynomial import Polynomial, polytrim


def clean_small_error(array: np.ndarray) -> np.ndarray:
    r""" clean relatively small quantity
    
    Args:
        array: target array
    
    Returns:
        cleaned array
    
    """

    def compare_and_clean(a: float, b: float) -> Tuple[float, float]:
        r""" discard tiny or relatively tiny real or imaginary parts of elements
        """
        if a == 0 or b == 0:
            return a, b

        a_abs = np.abs(a)
        b_abs = np.abs(b)

        abs_error = 10 ** (-2)
        rel_error = 10 ** (-4)

        if a_abs < abs_error and a_abs / b_abs < rel_error:
            return 0, b

        if b_abs < abs_error and b_abs / a_abs < rel_error:
            return a, 0

        return a, b

    for i in range(len(array)):
        real, imag = compare_and_clean(np.real(array[i]), np.imag(array[i]))
        array[i] = real + 1j * imag

    return array


def signal_unitary(x: float) -> np.ndarray:
    r"""signal unitary W(x)
    
    Args:
        x: variable x in [-1, 1]
        
    Returns:
        matrix W(x = x)
    
    """
    assert  -1 <= x <= 1, "x must be in domain [-1, 1]"
    return np.array([[x, 1j * np.sqrt(1 - x ** 2)], 
                     [1j * np.sqrt(1 - x ** 2), x]])


def second_condition_verification(P: Polynomial, k: int, error: float = 1e-6) -> bool:
    r"""verify whether the second condition of theorem 1 holds, and slightly decrease the error of Polynomial
    
    Args:
        P: polynomial P(x)
        k: parameter that determine parity
        error: tolerated error
        
    Returns:
        determine whether P has parity-(k mod 2)
    
    """
    parity = k % 2
    P_coef = P.coef
    
    for i in range(P.degree()):
        if i % 2 != parity:
            if np.abs(P_coef[i]) > error:
                return False
            P_coef[i] = 0  # this element should be 0
    return True


def third_condition_verification(P: Polynomial, Q: Polynomial, trials: int = 10, error: float = 1e-2) -> bool:
    r"""verify whether the thrid condition of theorem 1 holds
    
    Args:
        P: polynomial P(x)
        Q: polynomial Q(x)
        trials: number of tests
        error: tolerated error
        
    Returns:
        determine whether conditioin holds or not
    
    """
    P_conj = Polynomial(np.conj(P.coef))
    Q_conj = Polynomial(np.conj(Q.coef))

    test_poly = P * P_conj + Polynomial([1, 0, -1]) * Q * Q_conj
    for _ in range(trials):
        x = np.random.rand() * 2 - 1  # sample x from [-1, 1]
        y = test_poly(x)
        if abs(np.real(y) - 1) > error or abs(np.imag(y)) > error:
            print(np.real(y) - 1)
            return False
    return True


def equation_2_verification(phi: float, P: Polynomial, P_hat: Polynomial, Q_hat: Polynomial, 
                            trials: int = 10, error: float = 0.01) -> bool:
    r"""verify phi during the iteration of finding Phi
    
    Args:
        phi: rotation angle
        P: polynomial P(x)
        P_hat: updated polynomial
        Q_hat: updated polynomial
        trials: number of tests
        error: tolerated error
        
    Returns:
        determine whether the equation (2) for phi, P and P_hat & Q_hat holds
    
    """
    
    def block_encoding_P_hat(x):
        return np.array([[P_hat(x), 1j * Q_hat(x) * np.sqrt(1 - x ** 2)], 
                         [1j * np.conj(Q_hat(x)) * np.sqrt(1 - x ** 2), np.conj(P_hat(x))]])

    rz = np.array([[np.exp(1j * phi), 0], 
                   [0, np.exp(-1j * phi)]])
    
    for _ in range(trials):
        x = np.random.rand() * 2 - 1 # sample x from [-1, 1]
        matrix = np.matmul(np.matmul(block_encoding_P_hat(x), signal_unitary(x)), rz)
        y = matrix[0, 0]
        if np.abs(y - P(x)) > error:
            print(y)
            print(P(x))
            return False
    return True
    

def processing_unitary(matrix_phi: List[np.ndarray], x: float) -> np.ndarray:
    r"""processing unitary W_Phi(x)
    
    Args:
        matrix_Phi: array of phi's matirces
        x: variable x in [-1, 1]
        
    Returns:
        matrix W_Phi(x = x)
    
    """
    assert -1 <= x <= 1, "x must be in domain [-1, 1]"
    
    W = signal_unitary(x) 
    M = matrix_phi[0]
    for i in range(1, len(matrix_phi)):
        M = np.matmul(M, np.matmul(W, matrix_phi[i]))
    return M


def Phi_verification(Phi: np.ndarray, P: Polynomial, trials: int = 100, error: float = 1e-6) -> bool:
    r"""verify the final Phi
    
    Args:
        Phi: array of phi's
        P: polynomial P(x)
        trials: number of tests
        error: tolerated error
        
    Returns:
        determine whether W_Phi(x) is a block encoding of P(x)
    
    """
    def rz(theta: float) -> np.ndarray:
        return np.array([[np.exp(1j * theta), 0], 
                         [0, np.exp(-1j * theta)]])
        
    matrix_phi = list(map(rz, Phi))

    for _ in range(trials):
        x = np.random.rand() * 2 - 1 # sample x from [-1, 1]
        y = processing_unitary(matrix_phi, x)[0, 0]
        if np.abs(y - P(x)) > error:
            warnings.warn("Phi verification failed: " + 
                          f"y - P(x) = {y - P(x)}, error = {np.abs(y - P(x))}")
            return False
    return True


def update_polynomial(P: Polynomial, Q: Polynomial, phi: float) -> Tuple[Polynomial, Polynomial]:
    r"""update P, Q by given phi according to proof in theorem 1
    
    Args:
        P: polynomial P(x)
        Q: polynomial Q(x)
        phi: derived phi
        
    Returns:
        updated P(x), Q(x)
    
    """
    poly_1 = Polynomial([0, 1]) # x
    poly_2 = Polynomial([1, 0, -1]) # 1 - x^2
    
    # P = e^{-i phi} x P + e^{i phi} (1 - x^2) Q
    # Q = e^{i phi} x Q - e^{-i phi} P
    P_new = np.exp(-1j * phi) * poly_1 * P + np.exp(1j * phi) * poly_2 * Q
    Q_new = np.exp(1j * phi) * poly_1 * Q - np.exp(-1j * phi) * P

    # clean the error that is lower than 0.001
    P_new = Polynomial(polytrim(P_new.coef, 0.001))
    Q_new = Polynomial(polytrim(Q_new.coef, 0.001))
    
    # clean the error further, 
    P_new.coef = clean_small_error(P_new.coef)
    Q_new.coef = clean_small_error(Q_new.coef)
    
    if P_new.degree() >= P.degree() and np.abs(P_new.coef[-1]) < np.abs(P_new.coef[-3]):
        P_new.coef = np.delete(np.delete(P_new.coef, -1), -1)
    if Q_new.degree() >= Q.degree() > 0 and np.abs(Q_new.coef[-1]) < np.abs(Q_new.coef[-3]):
        Q_new.coef = np.delete(np.delete(Q_new.coef, -1), -1)
    
    # used for debug, can be removed in formal version    
    assert P_new.degree() < P.degree(), print(P_new, '\n', P)
    assert Q_new.degree() < Q.degree() or Q.degree() == 0, print(Q_new, '\n', Q)
    assert second_condition_verification(P_new, P.degree() - 1)
    assert second_condition_verification(Q_new, Q.degree() - 1)
    assert third_condition_verification(P_new, Q_new), print(P_new, '\n', Q_new)
    assert equation_2_verification(phi, P, P_new, Q_new)

    return P_new, Q_new
    

def alg_find_Phi(P: Polynomial, Q: Polynomial, k: int) -> np.ndarray:
    r"""The algorithm of finding phi's by theorem 1
    
    Args:
        P: polynomial P(x)
        Q: polynomial Q(x)
        k: length of returned array
        
    Returns:
        array of phi's
    
    """
    n = P.degree()
    m = Q.degree()
    
    # condition check for theorem 1 
    assert n <= k, "the condition for P's degree is not satisfied"
    assert m <= max(0, k - 1), "the condition for Q's degree is not satisfied"
    assert second_condition_verification(P, k), "the condition for P's parity is not satisfied"
    assert second_condition_verification(Q, k - 1), "the condition for Q's parity is not satisfied"
    assert third_condition_verification(P, Q), "the third equation for P, Q is not satisfied"
    
    i = k
    Phi = np.zeros([k + 1])

    while n > 0:
        # assign phi
        Phi[i] = (np.log(P.coef[n] / Q.coef[m]) * -1j / 2).real
        
        if Phi[i] == 0:
            Phi[i] = np.pi

        # update step
        P, Q = update_polynomial(P, Q, Phi[i])

        n = P.degree()
        m = Q.degree()
        i = i - 1
    
    for j in range(1, i):
        Phi[j] = (-1) ** (j - 1) * pi / 2
    Phi[0] = (-1j * np.log(P.coef[0])).real

    return Phi


def poly_A_hat_generation(P: Polynomial) -> Polynomial:
    r"""function for \hat{A} generation
    
    Args:
        P: polynomial P(x)
        
    Returns:
        polynomial \hat{A}(y) = 1 - P(x)P^*(x), with y = x^2
    
    """
    P_conj = Polynomial(np.conj(P.coef))
    A = 1 - P * P_conj
    A_coef = A.coef
    coef = [A_coef[0]]
    coef.extend(A_coef[2 * i] for i in range(1, P.degree() + 1))
    return Polynomial(np.array(coef))


def poly_A_hat_decomposition(A_hat: Polynomial, error: float = 0.001) -> Tuple[float, List[float]]:
    r"""function for \hat{A} generation
    
    Args:
        A_hat: polynomial \hat{A}(x)
        error: tolerated error
        
    Returns:
        Tuple: including the following elements
        - leading coefficient of \hat{A}
        - list of roots of \hat{A} such that there exist no two roots that are complex conjugates
    
    """
    leading_coef = A_hat.coef[A_hat.degree()]

    # remove one 1 and 0 (if k is even) from this list
    roots = [
        i
        for i in A_hat.roots()
        if (np.abs(np.real(i) - 1) >= error or np.abs(np.imag(i)) >= error)
        and np.abs(i) >= error
    ]

    # Note that root function in numpy return roots in complex conjugate pairs
    # Now elements in roots are all in pairs
    output_roots = [roots[i] for i in range(len(roots)) if i % 2 == 0]

    return leading_coef, output_roots


def poly_Q_generation(leading_coef: float, roots: List[float], k: int) -> Polynomial:
    r"""function for \hat{A} generation
    
    Args:
        leading_coef: leading coefficient of \hat{A}
        roots: filtered list of roots of \hat{A}
        k: parity that affects decomposition
        
    Returns:
        Tuple: including the following elements
        - leading coefficient of \hat{A}
        - list of roots of \hat{A} such that there exist no two roots that are complex conjugates
        
    """
    a = np.sqrt(-1 * leading_coef)

    Q = Polynomial([a])
    for item in roots:
        Q = Q * Polynomial([-1 * item, 0, 1])

    if k % 2 == 0:
        Q = Q * Polynomial([0, 1]) 
        return Q
    return Q
    

def alg_find_Q(P: Polynomial, k: int) -> Polynomial:
    r"""The algorithm of finding Q by theorem 2
    
    Args:
        P: polynomial P(x)
        k: length of returned array
        
    Returns:
        polynomial Q(x)
    
    """
    n = P.degree()
    
    # condition check for theorem 1 
    assert n <= k, "the condition for P's degree is not satisfied"
    assert second_condition_verification(P, k), "the condition for P's parity is not satisfied"
    
    A_hat = poly_A_hat_generation(P)
    leading_coef, roots = poly_A_hat_decomposition(A_hat)
    Q = poly_Q_generation(leading_coef, roots, k)
     
    return Q


def quantum_signal_processing(P: Polynomial, k: int = None) -> np.ndarray:
    r""" Compute Phi that transfer a block encoding of x to a block encoding of P(x) by W_\Phi(x)
    
    Args:
        P: polynomial P(x)
        k: length of returned array
        
    Returns:
        array of phi's
    
    """
    if k is None:
        k = P.degree()
    
    Q = alg_find_Q(P, k)
    Phi = alg_find_Phi(P, Q, k)
    
    # Phi_verification(Phi, P)
        
    return Phi

def reflection_based_quantum_signal_processing(P: Polynomial) -> np.ndarray:
    r""" Compute Phi that transfer a block encoding of x to a block encoding of P(x) with R_\Phi(x)
    
    Args:
        P: polynomial P(x)
        
    Returns:
        array of phi's
    
    """
    Phi = quantum_signal_processing(P)
    k = P.degree()
    Phi_new = np.zeros([k])
    
    Phi_new[0] = Phi[0] + Phi[k] + (k - 1) * np.pi / 2
    for i in range(1, k):
        Phi_new[i] = Phi[i] - np.pi / 2
    
    # assertion
    phi_sum = 0
    phi_alternate = 0
    for i in range(k):
        phi_sum += Phi_new[i]
        phi_alternate += ((-1) ** i) * Phi_new[i]
        
    assert np.abs(P(1) - np.exp(1j * phi_sum)) < 10 ** (-8)
    assert np.abs(P(-1) - ((-1) ** k) * np.exp(1j * phi_sum)) < 10 ** (-8)
    if k % 2 == 0:
        assert np.abs(P(0) - np.exp(-1j * phi_alternate)) < 10 ** (-8)
    
    return Phi_new
