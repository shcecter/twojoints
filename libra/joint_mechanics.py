import numpy as np


def get_lmt_from_q(op, ip, q):
    '''
    op - origin point
    ip - insertion point
    q - angle

    Returns: distance between op and ip given angle q
    '''
    lmt = np.sqrt(op**2 + ip**2 - 2*op*ip*np.cos(q))
    return lmt


def get_arm(op, ip, q):
    '''
    op - origin point
    ip - insertion point
    q = angle

    Returns: muscle arm given angle q
    '''
    arm = op*ip*np.sin(q) / get_lmt_from_q(op, ip, q)
    return arm


def get_muscle_dynamics(muscle, lm, lmt, t):
    '''
    muscle - libra.muscle.Thelen2003 instance
    lm - muscle (contractile part only) length
    lmt - muscle + tendon length
    t - current time step (during simulation)
    Returns: muscle velocity and muscle se force: d_lm, force_se
    '''
    lm0     = muscle.S['lm0']
    ltslack = muscle.P['ltslack']
    alpha0  = muscle.P['alpha0']
    lmopt   = muscle.P['lmopt']
    vmmax   = muscle.P['vmmax']
    fm0     = muscle.P['fm0']

    lmt0 = lmt
    d_lm = muscle.vm_eq(t, lm, lm0, lmt0, lmopt, ltslack, alpha0, vmmax, fm0)

    alpha = muscle.penn_ang(lmt, lm, lm0, alpha0)
    lt = lmt - lm*np.cos(alpha)
    force_se = muscle.force_se(lt) * fm0
    # second approach to calc forces
    '''fl = muscle.force_l(lm)
    a = muscle.activation(t)
    fvm = muscle.force_vm(d_lm, a, fl)
    force_se = (a*fl*fvm + muscle.force_pe(lm))*np.cos(alpha)*fm0'''
    return d_lm, force_se


def calc_muscle_force(mus, mus_idx, q_idx, sfo, sfi, sol, time):
    _lm = sol[:, mus_idx]
    _lmt = get_lmt_from_q(sfo, sfi, sol[:, q_idx])
    alpha = mus.penn_ang(_lmt, _lm, mus.P['lmopt'], mus.P['alpha0'])
    lt = _lmt - _lm*np.cos(alpha)
    _force = [mus.force_se(lt[i]) for i in range(len(time))]
    return _force


def two_j_mech_modfed(t, y):
    '''
    just wrapper to use `two_joint_mech` with scipy.solve_ivp
    '''
    return two_joint_mech(y, t, ps, muscles)


def two_joint_mech(state, t, ps, muscles):
    '''
    func being used in scipy.odeint to simulate two joint pendulum,
    muscle forces calculated with 'get_muscle_dynamics' func and adding as
    generalized coordinates
    state - two angles, angle speeds, muscle lengths;
    t - current time step;
    ps - tuple with origin and insertion values for muscles;
    muscle - tuple with Thelen2003 objects with custom parameters and states;

    Returns: right side of two joint mechanics equation
    [d_q1, d_q2, dd_q1, dd_q2, d_sf_lm, d_se_lm, d_ef_lm]
    '''
    q1, q2, d_q1, d_q2, sf_lm, se_lm, ef_lm = state
    l1, l2 = .3, .25
    m1, m2 = 1.8, 1.6  # m1=2.7
    r1, r2 = l1/2, l2/2
    i1, i2 = m1*l1**2/3, m2*l1**2/3
    g = 9.81

    (sfo, sfi,
     seo, sei,
     efo, efi) = ps

    (sf_muscle, se_muscle, ef_muscle) = muscles

    sf_lmt = get_lmt_from_q(sfo, sfi, np.pi - q1)
    se_lmt = get_lmt_from_q(seo, sei, q1)
    ef_lmt = get_lmt_from_q(efo, efi, np.pi - q2)

    d_sf_lm, sf_force_se = get_muscle_dynamics(sf_muscle, sf_lm, sf_lmt, t)
    d_se_lm, se_force_se = get_muscle_dynamics(se_muscle, se_lm, se_lmt, t)
    d_ef_lm, ef_force_se = get_muscle_dynamics(ef_muscle, ef_lm, ef_lmt, t)
    T_sf = sf_force_se * get_arm(sfo, sfi, np.pi - q1)
    T_se = se_force_se * get_arm(seo, sei, q1)
    T_ef = ef_force_se * get_arm(efo, efi, np.pi - q2)

    # constraints by friction
    Qfr1 = 0.
    Qfr2 = 0.
    if q1 <= np.deg2rad(15) and d_q1 < 0.:
        Qfr1 = -d_q1 * 200
    if q1 >= np.deg2rad(160) and d_q1 > 0.:
        Qfr1 = -d_q1 * 100

    if q2 <= np.deg2rad(15) and d_q2 < 0.:
        Qfr2 = -d_q2 * 100
    if q2 >= np.deg2rad(160) and d_q2 > 0.:
        Qfr2 = -d_q2 * 100

    Q1 = T_sf - T_se + Qfr1
    Q2 = T_ef + Qfr2

    Q = np.matrix([[Q1], [Q2]])

    M11 = i1 + i2 + (m1*r1**2) + m2*(l1**2 + r2**2 + 2*l1*r2*np.cos(q2))
    M12 = i2 + m2*(r2**2 + l1*r2*np.cos(q2))
    M21 = M12
    M22 = i2 + (m2*r2**2)
    M = np.matrix([[M11, M12], [M21, M22]])
    C1 = -m2*l1*d_q2**2*r2*np.sin(q2) - 2*m2*l1*d_q1*d_q2*r2*np.sin(q2)
    C2 = m2*l1*d_q1**2*r2*np.sin(q2)
    C = np.matrix([[C1], [C2]])
    G1 = g*np.sin(q1)*(m2*l1 + m1*r1) + g*m2*r2*np.sin(q1 + q2)
    G2 = g*m2*r2*np.sin(q1 + q2)
    G = np.matrix([[G1], [G2]])
    acc = np.linalg.inv(M) * (Q - C - G)
    dd_q1, dd_q2 = acc[0, 0], acc[1, 0]

    return [d_q1, d_q2, dd_q1, dd_q2, d_sf_lm, d_se_lm, d_ef_lm]
