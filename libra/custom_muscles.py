import numpy as np

from .joint_mechanics import get_lmt_from_q
from .muscles import Thelen2003


def create_custom_muscle(lmt0, lm0, lt0, ltslack, lmopt, fm0):
    '''
    args : custom muscle parameters and states, listed below in this module
    Returns : muscle object with args parameters and states
    '''
    muscle = Thelen2003()
    muscle.set_states('libra/data/muscle_states.txt')
    muscle.set_parameters('libra/data/muscle_parameters.txt')
    muscle.S['lmt0'] = lmt0
    muscle.S['lm0'] = lm0
    muscle.S['lt0'] = lt0
    muscle.P['ltslack'] = ltslack
    muscle.P['lmopt'] = lmopt
    muscle.P['fm0'] = fm0
    return muscle


init_q = np.deg2rad(15)
stop_q = np.deg2rad(85)

# shoulder flexor muscle parameters
sfo = 0.02
sfi = 0.1
sf_init_lmt_ = get_lmt_from_q(sfo, sfi, np.pi - init_q)
sf_stop_lmt_ = get_lmt_from_q(sfo, sfi, np.pi - stop_q)
sf_ltslack = sf_lt0 = 0.01  # choosen arbitrary from init|stop lmt
sf_lmt0 = sf_init_lmt_
sf_lm0 = sf_lmt0 - sf_lt0
sf_init_lm_ = sf_init_lmt_ - sf_ltslack
sf_stop_lm_ = sf_init_lm_ - sf_ltslack
sf_lmopt = (sf_init_lm_ + sf_stop_lm_) / 2
sf_fm0 = 2000
sf_vars = [sfo, sfi, sf_lmt0, sf_lm0, sf_lt0, sf_ltslack, sf_lmopt, sf_fm0]

# shoulder extensor muscle parameters estimation
seo = 0.04  # 0.08
sei = 0.3 / 2
se_init_lmt_ = get_lmt_from_q(seo, sei, init_q)
se_stop_lmt_ = get_lmt_from_q(seo, sei, stop_q)
se_ltslack = se_lt0 = 0.01  # arbitrary
se_lmt0 = se_init_lmt_
se_lm0 = se_lmt0 - se_lt0
se_init_lm_ = se_init_lmt_ - se_ltslack
se_stop_lm_ = se_stop_lmt_ - se_ltslack
se_lmopt = (se_init_lm_ + se_stop_lm_) / 2
se_fm0 = 2000
se_vars = [seo, sei, se_lmt0, se_lm0, se_lt0, se_ltslack, se_lmopt, se_fm0]

# elbow flexor muscle parameters estimation
efo = 0.1
efi = 0.04
ef_init_lmt_ = get_lmt_from_q(efo, efi, np.pi - init_q)
ef_stop_lmt_ = get_lmt_from_q(efo, efi, np.pi - stop_q)
ef_ltslack = ef_lt0 = 0.03  # arbitrary
ef_lmt0 = ef_init_lmt_
ef_lm0 = ef_lmt0 - ef_lt0
ef_init_lm_ = ef_init_lmt_ - ef_ltslack
ef_stop_lm_ = ef_stop_lmt_ - ef_ltslack
ef_lmopt = (ef_init_lm_ + ef_stop_lm_) / 2
ef_fm0 = 100
ef_vars = [efo, efi, ef_lmt0, ef_lm0, ef_lt0, ef_ltslack, ef_lmopt, ef_fm0]

if __name__ == '__main__':
    print('Shoulder flexor params')
    print(sf_vars)
    print('Shoulder extensor params')
    print(se_vars)
    print('Elbow flexor params')
    print(ef_vars)
