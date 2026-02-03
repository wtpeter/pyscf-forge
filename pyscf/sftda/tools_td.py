#/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

import numpy as np
from functools import reduce
from pyscf import lib, scf, dft
from pyscf.scf.uhf import spin_square as spin_square_scf
from pyscf.data.nist import HARTREE2EV
from pyscf.lib import logger

def spin_square(td, state=None):
    r'''calculator of <S^2> of excited states using tddft/tda.
        Ref. J. Chem. Phys. 2011, 134, 134101.
    '''
    mf = td._scf
    s20, _ = mf.spin_square()
    sz = mf.mol.spin / 2.0

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = mo_occ[0] == 0
    viridxb = mo_occ[1] == 0
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:, viridxa]
    orbvb = mo_coeff[1][:, viridxb]

    ovlp = mf.get_ovlp()
    sab_oo = orboa.conj().T @ ovlp @ orbob
    sba_oo = sab_oo.conj().T
    sab_vo = orbva.conj().T @ ovlp @ orbob
    sba_ov = sab_vo.conj().T
    sba_vo = orbvb.conj().T @ ovlp @ orboa
    sab_ov = sba_vo.conj().T

    if state is None:
        state = np.arange(td.nstates)
    if isinstance(state, int):
        states = [state]
    else:
        states = state
    states = np.array(states)
    xs = np.array([td.xy[i][0].T for i in states])
    if isinstance(td.xy[0][1], np.ndarray):
        ys = np.array([td.xy[i][1].T for i in states])
    else:
        ys = None

    if td.extype==0:
        assert xs[0].shape==sab_vo.shape
        P_ab = lib.einsum('nai,naj,jk,ki->n', xs.conj(), xs, sba_oo, sab_oo) \
               - lib.einsum('nai,nbi,kb,ak->n', xs.conj(), xs, sba_ov, sab_vo) \
               + lib.einsum('nai,nbj,jb,ai->n', xs.conj(), xs, sba_ov, sab_vo)
        if ys is not None:
            assert ys[0].shape==sba_vo.shape
            P_ab += lib.einsum('nai,naj,ik,kj->n', ys.conj(), ys, sab_oo, sba_oo) \
                    - lib.einsum('nai,nbi,ka,bk->n', ys.conj(), ys, sab_ov, sba_vo) \
                    + lib.einsum('nai,nbj,ia,bj->n', ys.conj(), ys, sab_ov, sba_vo) \
                    - 2 * lib.einsum('nai,nbj,ai,bj->n', xs.conj(), ys, sab_vo, sba_vo).real
        ds2 = P_ab + 2 * sz + 1
    elif td.extype==1:
        assert xs[0].shape==sba_vo.shape
        P_ab = lib.einsum('nai,naj,jk,ki->n', xs.conj(), xs, sab_oo, sba_oo) \
               - lib.einsum('nai,nbi,kb,ak->n', xs.conj(), xs, sab_ov, sba_vo) \
               + lib.einsum('nai,nbj,jb,ai->n', xs.conj(), xs, sab_ov, sba_vo)
        if ys is not None:
            assert ys[0].shape==sab_vo.shape
            P_ab += lib.einsum('nai,naj,ik,kj->n', ys.conj(), ys, sba_oo, sab_oo) \
                    - lib.einsum('nai,nbi,ka,bk->n', ys.conj(), ys, sba_ov, sab_vo) \
                    + lib.einsum('nai,nbj,ia,bj->n', ys.conj(), ys, sba_ov, sab_vo) \
                    - 2 * lib.einsum('nai,nbj,ai,bj->n', xs.conj(), ys, sba_vo, sab_vo).real
        ds2 = P_ab - 2 * sz + 1
    
    s2s = s20 + ds2.real
    if isinstance(state, int):
        return s2s[0]
    else:
        return s2s

def extract_state(td, Smin=0.0, Smax=0.7, verbose=None):
    if verbose is None:
        verbose = td.verbose
    log = logger.new_logger(td, verbose=verbose)

    s2 = spin_square(td, np.arange(td.nstates))
    targets = np.where((s2 >= Smin) & (s2 <= Smax))[0]
    for i in targets:
        msg = f'State {i+1:>2d}: E = {td.e[i]*HARTREE2EV:>7.4f} eV, <S^2> = {s2[i]:>6.3f}'
        if log.verbose >= logger.INFO:
            msg += ', dominated by:'
            x = td.xy[i][0]
            flat_indices = np.argsort(abs(x).ravel())[-2:][::-1]
            for o, v in zip(*np.unravel_index(flat_indices, x.shape)):
                if td.extype==0:
                    nocca = td._scf.mo_occ[0].sum()
                    msg += f' {x[o, v]:>6.3f}@({int(o+1)}b, {int(v+nocca+1)}a)'
                elif td.extype==1:
                    noccb = td._scf.mo_occ[1].sum()
                    msg += f' {x[o, v]:>6.3f}@({int(o+1)}a, {int(v+noccb+1)}b)'
        log.note(msg)
    return targets
