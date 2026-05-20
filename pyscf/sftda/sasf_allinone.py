#!/usr/bin/env python
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
#
# Spin-Adapted TDA
#
# Author: Tai Wang <wtpeter@pku.edu.cn>
#

import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf import dft
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.tdscf.uhf import TDBase
from pyscf.dft.gen_grid import NBINS
from pyscf import __config__
from pyscf.dft.numint import _scale_ao_sparse, _dot_ao_ao_sparse, _dot_ao_dm_sparse, _contract_rho_sparse
from pyscf.tdscf._lr_eig import eigh as lr_eigh

def nr_rks_fxc1_gga(ni, mol, grids, xc_code, dms, fxc, max_memory=2000):
    nset = dms.shape[0]
    vmat = np.zeros_like(dms)

    nao = mol.nao_nr()
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)

    aow = None
    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, 1, max_memory=max_memory):
        p0, p1 = p1, p1 + weight.size
        _fxc = fxc[:, :, p0:p1] * weight
        for num in range(nset):
            dm = dms[num]
            dm = np.asarray(dms[num], order='C')
            ngrids = ao.shape[1]

            c0 = _dot_ao_dm_sparse(ao[0], dm, nbins, mask, pair_mask, ao_loc)
            rho0 = _contract_rho_sparse(ao[0], c0, mask, ao_loc)
            R_grad = np.empty((3, ngrids))
            for j in range(1, 4):
                R_grad[j - 1] = _contract_rho_sparse(ao[j], c0, mask, ao_loc)
            c_grad = []
            for i in range(1, 4):
                c_grad.append(_dot_ao_dm_sparse(ao[i], dm, nbins, mask, pair_mask, ao_loc))
            L_grad = np.empty((3, ngrids))
            for i in range(1, 4):
                L_grad[i - 1] = _contract_rho_sparse(ao[0], c_grad[i - 1], mask, ao_loc)
            tau = np.empty((3, 3, ngrids))
            for i in range(1, 4):
                for j in range(1, 4):
                    tau[i - 1, j - 1] = _contract_rho_sparse(ao[j], c_grad[i - 1], mask, ao_loc)

            U = np.zeros((4, 4, weight.size))
            U[0, 0] = _fxc[0, 0] * rho0
            U[0, 0] += lib.einsum('ig,ig->g', _fxc[1:4, 0], L_grad)  # sum_i f_{i0} L_i
            U[0, 0] += lib.einsum('jg,jg->g', _fxc[0, 1:4], R_grad)  # sum_j f_{0j} R_j
            U[0, 0] += lib.einsum('ijg,ijg->g', _fxc[1:4, 1:4], tau)
            U[1:4, 0] += _fxc[1:4, 0] * rho0
            U[1:4, 0] += lib.einsum('ijg,jg->ig', _fxc[1:4, 1:4], R_grad)
            U[0, 1:4] += _fxc[0, 1:4] * rho0
            U[0, 1:4] += lib.einsum('ijg,ig->jg', _fxc[1:4, 1:4], L_grad)
            U[1:4, 1:4] += _fxc[1:4, 1:4] * rho0

            for i in range(4):
                wv_i = np.asarray(U[i, :, :], order='C')
                aow = _scale_ao_sparse(ao, wv_i, mask, ao_loc)
                v_chunk = _dot_ao_ao_sparse(ao[i], aow, None, nbins, mask, pair_mask, ao_loc, hermi=0, out=None)
                vmat[num] += v_chunk
    return vmat


def nr_rks_fxc1_mgga(ni, mol, grids, xc_code, dms, fxc, max_memory=2000):
    nset = dms.shape[0]
    vmat = np.zeros_like(dms)

    nao = mol.nao_nr()
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2

    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)

    aow = None
    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, 1, max_memory=max_memory):
        p0, p1 = p1, p1 + weight.size
        _fxc = fxc[:, :, p0:p1] * weight
        for num in range(nset):
            dm = np.asarray(dms[num], order='C')
            ngrids = ao.shape[1]
            c0 = _dot_ao_dm_sparse(ao[0], dm, nbins, mask, pair_mask, ao_loc)
            rho0 = _contract_rho_sparse(ao[0], c0, mask, ao_loc)
            R_grad = np.empty((3, ngrids))
            for j in range(1, 4):
                R_grad[j - 1] = _contract_rho_sparse(ao[j], c0, mask, ao_loc)
            c_grad = []
            for i in range(1, 4):
                c_grad.append(_dot_ao_dm_sparse(ao[i], dm, nbins, mask, pair_mask, ao_loc))
            L_grad = np.empty((3, ngrids))
            for i in range(1, 4):
                L_grad[i - 1] = _contract_rho_sparse(ao[0], c_grad[i - 1], mask, ao_loc)
            tau = np.empty((3, 3, ngrids))
            for i in range(1, 4):
                for j in range(1, 4):
                    tau[i - 1, j - 1] = _contract_rho_sparse(ao[j], c_grad[i - 1], mask, ao_loc)

            U = np.zeros((4, 4, weight.size))
            U[0, 0] = _fxc[0, 0] * rho0
            U[0, 0] += lib.einsum('ig,ig->g', _fxc[1:4, 0], L_grad)
            U[0, 0] += lib.einsum('jg,jg->g', _fxc[0, 1:4], R_grad)
            U[0, 0] += lib.einsum('ijg,ijg->g', _fxc[1:4, 1:4], tau)
            U[1:4, 0] = _fxc[1:4, 0] * rho0
            U[1:4, 0] += lib.einsum('ijg,jg->ig', _fxc[1:4, 1:4], R_grad)
            U[1:4, 0] += 0.5 * _fxc[4, 0].reshape(1, -1) * L_grad
            U[1:4, 0] += 0.5 * lib.einsum('jg,ijg->ig', _fxc[4, 1:4], tau)
            U[0, 1:4] = _fxc[0, 1:4] * rho0
            U[0, 1:4] += lib.einsum('ijg,ig->jg', _fxc[1:4, 1:4], L_grad)
            U[0, 1:4] += 0.5 * _fxc[0, 4].reshape(1, -1) * R_grad
            U[0, 1:4] += 0.5 * lib.einsum('ig,ijg->jg', _fxc[1:4, 4], tau)
            U[1:4, 1:4] = _fxc[1:4, 1:4] * rho0
            U[1:4, 1:4] += 0.5 * lib.einsum('ig,jg->ijg', _fxc[1:4, 4], R_grad)
            U[1:4, 1:4] += 0.5 * lib.einsum('jg,ig->ijg', _fxc[4, 1:4], L_grad)
            U[1:4, 1:4] += 0.25 * _fxc[4, 4].reshape(1, 1, -1) * tau

            for i in range(4):
                wv_i = np.asarray(U[i, :, :], order='C')
                aow = _scale_ao_sparse(ao, wv_i, mask, ao_loc)
                v_chunk = _dot_ao_ao_sparse(ao[i], aow, None, nbins, mask, pair_mask, ao_loc, hermi=0, out=None)
                vmat[num] += v_chunk

    return vmat

def gen_rohf_response_sc(mf, mo_coeff=None, mo_occ=None, hermi=0, max_memory=None):
    '''
    response function for Sf=Si
    '''
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mol = mf.mol
    assert isinstance(mf, dft.roks.ROKS) or isinstance(mf, dft.rks_symm.SymAdaptedROKS)

    s = (mol.nelec[0] - mol.nelec[1]) * 0.5
    assert s >= 0.5, 'SATDA for Sf=Si only supports case that Si>=1/2.'

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    xctype = ni._xc_type(mf.xc)
    if xctype != 'HF':
        fxc_d0 = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]  # TODO: needed to be checked
        fxc_ref = 0.5 * (fxc_d0[0, :, 0] - fxc_d0[0, :, 1] - fxc_d0[1, :, 0] + fxc_d0[1, :, 1])
        umf = mf.to_uks()
        uni = umf._numint
        _, _, fxc = uni.cache_xc_kernel(mol, mf.grids, mf.xc, umf.mo_coeff, umf.mo_occ, 1)
        fxc_s = 0.5 * (fxc[0, :, 0] + fxc[0, :, 1] + fxc[1, :, 0] + fxc[1, :, 1])
        fxc_cv0cv = 0.5 * (fxc[0, :, 0] + fxc[0, :, 1] - fxc[1, :, 0] - fxc[1, :, 1])
        fxc_cvcv0 = 0.5 * (fxc[0, :, 0] - fxc[0, :, 1] + fxc[1, :, 0] - fxc[1, :, 1])
        fxc_cv0co = fxc[0, :, 1] + fxc[1, :, 1]
        fxc_cv0ov = fxc[0, :, 0] + fxc[1, :, 0]
        fxc_cocv0 = fxc[1, :, 0] + fxc[1, :, 1]
        fxc_ovcv0 = fxc[0, :, 0] + fxc[0, :, 1]

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

    if mf.do_nlc():
        logger.warn(mf, "NLC contribution in gen_response is NOT included")

    def vind(dms_co, dms_cv, dms_ov, dms_cv0):
        n_co = len(dms_co)
        n_cv = len(dms_cv)
        n_ov = len(dms_ov)
        idx1 = n_co
        idx2 = idx1 + n_cv
        idx3 = idx2 + n_ov

        v1ao_co = np.zeros_like(dms_co)
        v1ao_cv = np.zeros_like(dms_cv)
        v1ao_ov = np.zeros_like(dms_ov)
        v1ao_cv0 = np.zeros_like(dms_cv0)

        dms0 = np.concatenate((dms_co, dms_cv, dms_ov), axis=0)
        dms1 = np.concatenate((dms_co, dms_ov), axis=0)

        # Coulomb part
        dms_j = np.concatenate((dms_co, dms_ov, dms_cv0), axis=0)
        vcoul = mf.get_j(mol, dms_j, hermi)
        vcoul_co = vcoul[:idx1]
        vcoul_ov = vcoul[idx1:idx1+n_ov]
        vcoul_cv0 = vcoul[idx1+n_ov:]
        v1ao_co += np.sqrt(2) * vcoul_cv0
        v1ao_ov -= np.sqrt(2) * vcoul_cv0
        v1ao_cv0 += np.sqrt(2) * vcoul_co - np.sqrt(2) * vcoul_ov + 2 * vcoul_cv0

        # HF part
        if hybrid:
            dms = np.concatenate((dms_co, dms_cv, dms_ov, dms_cv0), axis=0)
            vk = mf.get_k(mol, dms, hermi) * hyb
            vj = vcoul[:idx1+n_ov] * hyb
            if omega != 0:
                vk += mf.get_k(mol, dms, hermi, omega=omega) * (alpha - hyb)
                vj += mf.get_j(mol, dms_j[:idx1+n_ov], hermi, omega=omega) * (alpha - hyb)
            vk_co = vk[:idx1]
            vk_cv = vk[idx1:idx2]
            vk_ov = vk[idx2:idx3]
            vk_cv0 = vk[idx3:]
            vj_co = vj[:idx1]
            vj_ov = vj[idx1:]
            v1ao_co += - vk_co + vj_co - np.sqrt((s + 1) / 2 / s) * vk_cv - vj_ov - np.sqrt(0.5) * vk_cv0
            v1ao_cv += - np.sqrt((s + 1) / 2 / s) * vk_co - vk_cv - np.sqrt((s + 1) / 2 / s) * vk_ov
            v1ao_ov += - vj_co - np.sqrt((s + 1) / 2 / s) * vk_cv + vj_ov - vk_ov + np.sqrt(0.5) * vk_cv0
            v1ao_cv0 += - np.sqrt(0.5) * vk_co + np.sqrt(0.5) * vk_ov - vk_cv0

        # K^Ref part
        if xctype != 'HF':
            vref0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms0, 0, hermi, None, None, fxc_ref, max_memory=max_memory)
            if xctype == 'LDA':
                vref1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms1, 0, hermi, None, None, fxc_ref, max_memory=max_memory)
            elif xctype =='GGA':
                vref1 = nr_rks_fxc1_gga(ni, mol, mf.grids, mf.xc, dms1, fxc_ref, max_memory=max_memory)
            elif xctype == 'MGGA':
                vref1 = nr_rks_fxc1_mgga(ni, mol, mf.grids, mf.xc, dms1, fxc_ref, max_memory=max_memory)
        else:
            vref0 = np.zeros_like(dms0)
            vref1 = np.zeros_like(dms1)
        vref0_co = vref0[:idx1]
        vref0_cv = vref0[idx1:idx2]
        vref0_ov = vref0[idx2:]
        vref1_co = vref1[:idx1]
        vref1_ov = vref1[idx1:]

        v1ao_co += vref0_co - vref1_co + np.sqrt((s + 1) / 2 / s) * vref0_cv + vref1_ov
        v1ao_cv += np.sqrt((s + 1) / 2 / s) * vref0_co + vref0_cv + np.sqrt((s + 1) / 2 / s) * vref0_ov
        v1ao_ov += vref1_co + np.sqrt((s + 1) / 2 / s) * vref0_cv - vref1_ov + vref0_ov

        # K^CV0 part
        if xctype != 'HF':
            v_cocv0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv0, 0, hermi, None, None, fxc_cocv0, max_memory=max_memory)
            v_cvcv0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv0, 0, hermi, None, None, fxc_cvcv0, max_memory=max_memory)
            v_ovcv0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv0, 0, hermi, None, None, fxc_ovcv0, max_memory=max_memory)
            v_cv0co = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_co, 0, hermi, None, None, fxc_cv0co, max_memory=max_memory)
            v_cv0cv = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv, 0, hermi, None, None, fxc_cv0cv, max_memory=max_memory)
            v_cv0ov = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_ov, 0, hermi, None, None, fxc_cv0ov, max_memory=max_memory)
            v_cv0cv0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv0, 0, hermi, None, None, fxc_s, max_memory=max_memory)
            v1ao_co += np.sqrt(0.5) * v_cocv0
            v1ao_cv -= np.sqrt((s + 1) / s) * v_cvcv0
            v1ao_ov -= np.sqrt(0.5) * v_ovcv0
            v1ao_cv0 += np.sqrt(0.5) * v_cv0co - np.sqrt((s + 1) / s) * v_cv0cv - np.sqrt(0.5) * v_cv0ov + v_cv0cv0
        return v1ao_co, v1ao_cv, v1ao_ov, v1ao_cv0

    orbos = mo_coeff[:, np.where(mo_occ == 1)[0]]
    dmoo = orbos @ orbos.T
    if xctype != 'HF':
        delta = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dmoo, 0, 1, None, None, fxc_ref, max_memory=max_memory)
    else:
        delta = np.zeros_like(dmoo)
    if hybrid:
        delta -= mf.get_k(mol, dmoo, 1) * hyb
        if omega != 0:
            delta -= mf.get_k(mol, dmoo, 1, omega=omega) * (alpha - hyb)
    return vind, 0.5 * delta

def gen_rohf_response_sf(mf, mo_coeff=None, mo_occ=None, hermi=0, max_memory=None):
    '''
    response function for Sf=Si-1
    '''
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mol = mf.mol
    assert isinstance(mf, dft.roks.ROKS) or isinstance(mf, dft.rks_symm.SymAdaptedROKS)

    s = (mol.nelec[0] - mol.nelec[1]) * 0.5
    assert s >= 1, 'SATDA for Sf=Si-1 only supports case that Si>=1.'

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    xctype = ni._xc_type(mf.xc)

    if xctype != 'HF':
        fxc_d0 = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]  # TODO: needed to be checked
        fxc_ref = 0.5 * (fxc_d0[0, :, 0] - fxc_d0[0, :, 1] - fxc_d0[1, :, 0] + fxc_d0[1, :, 1])

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

    if mf.do_nlc():
        logger.warn(mf, "NLC contribution in gen_response is NOT included")

    def vind(dms_co, dms_cv, dms_oo, dms_ov):
        n_co = len(dms_co)
        n_cv = len(dms_cv)
        n_oo = len(dms_oo)
        n_ov = len(dms_ov)
        idx1 = n_co
        idx2 = n_co + n_cv
        idx3 = n_co + n_cv + n_oo

        v1ao_co = np.zeros_like(dms_co)
        v1ao_cv = np.zeros_like(dms_cv)
        v1ao_oo = np.zeros_like(dms_oo)
        v1ao_ov = np.zeros_like(dms_ov)

        dms0 = np.concatenate((dms_co, dms_cv, dms_oo, dms_ov), axis=0)
        dms1 = np.concatenate((dms_co, dms_ov), axis=0)

        if xctype != 'HF':
            vref0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms0, 0, hermi, None, None, fxc_ref, max_memory=max_memory)
            if xctype == 'LDA':
                vref1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms1, 0, hermi, None, None, fxc_ref, max_memory=max_memory)
            elif xctype =='GGA':
                vref1 = nr_rks_fxc1_gga(ni, mol, mf.grids, mf.xc, dms1, fxc_ref, max_memory=max_memory)
            elif xctype == 'MGGA':
                vref1 = nr_rks_fxc1_mgga(ni, mol, mf.grids, mf.xc, dms1, fxc_ref, max_memory=max_memory)
        else:
            vref0 = np.zeros_like(dms0)
            vref1 = np.zeros_like(dms1)

        # HF part
        if hybrid:
            vk = mf.get_k(mol, dms0, hermi) * hyb
            vj = mf.get_j(mol, dms1, hermi) * hyb
            if omega != 0:
                vk += mf.get_k(mol, dms0, hermi, omega=omega) * (alpha - hyb)
                vj += mf.get_j(mol, dms1, hermi, omega=omega) * (alpha - hyb)
            vref0 -= vk
            vref1 -= vj
        vref0_co = vref0[:idx1]
        vref0_cv = vref0[idx1:idx2]
        vref0_oo = vref0[idx2:idx3]
        vref0_ov = vref0[idx3:]
        vref1_co = vref1[:idx1]
        vref1_ov = vref1[idx1:]

        v1ao_co += vref0_co + vref1_co / (2 * s - 1) + np.sqrt((2 * s + 1) / 2 / s) * vref0_cv
        v1ao_co += np.sqrt(2 * s / (2 * s - 1)) * vref0_oo + 2 * s / (2 * s - 1) * vref0_ov - vref1_ov / (2 * s - 1)
        v1ao_cv += vref0_co * np.sqrt((2 * s + 1) / 2 / s) + vref0_cv + np.sqrt((2 * s + 1) / (2 * s - 1)) * vref0_oo
        v1ao_cv += np.sqrt((2 * s + 1) / 2 / s) * vref0_ov
        v1ao_oo += np.sqrt(2 * s / (2 * s - 1)) * vref0_co + np.sqrt((2 * s + 1) / (2 * s - 1)) * vref0_cv
        v1ao_oo += vref0_oo + np.sqrt(2 * s / (2 * s - 1)) * vref0_ov
        v1ao_ov += 2 * s / (2 * s - 1) * vref0_co - vref1_co / (2 * s - 1) + np.sqrt((2 * s + 1) / 2 / s) * vref0_cv
        v1ao_ov += np.sqrt(2 * s / (2 * s - 1)) * vref0_oo + vref0_ov + vref1_ov / (2 * s - 1)

        return v1ao_co, v1ao_cv, v1ao_oo, v1ao_ov

    orbos = mo_coeff[:, np.where(mo_occ == 1)[0]]
    dmoo = orbos @ orbos.T
    if xctype != 'HF':
        delta = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dmoo, 0, 1, None, None, fxc_ref, max_memory=max_memory)
    else:
        delta = np.zeros_like(dmoo)
    if hybrid:
        delta -= mf.get_k(mol, dmoo, 1) * hyb
        if omega != 0:
            delta -= mf.get_k(mol, dmoo, 1, omega=omega) * (alpha - hyb)
    return vind, 0.5 * delta

def gen_vind_sc(td):
    mf = td._scf
    mo_coeff = mf.mo_coeff
    assert mo_coeff[0].dtype == np.double
    mo_occ = mf.mo_occ

    mol = mf.mol
    s = (mol.nelec[0] - mol.nelec[1]) * 0.5
    assert s >= 0.5, 'SASFTDA only supports case that Sf=Si>=1/2.'

    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]
    orbcs = mo_coeff[:, csidx]
    orbos = mo_coeff[:, osidx]
    orbvs = mo_coeff[:, vsidx]
    ncs = orbcs.shape[1]
    nos = orbos.shape[1]
    nvs = orbvs.shape[1]
    idx1 = ncs * nos
    idx2 = idx1 + ncs * nvs
    idx3 = idx2 + 1
    idx4 = idx3 + nos * nvs

    vresp, fockz = gen_rohf_response_sc(mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0, max_memory=td.max_memory)

    fock = mf.get_fock()
    focka = fock.focka
    fockb = fock.fockb
    fock0 = focka - fockz

    fock_coco1 = orbos.T @ (fock0 - fockz) @ orbos
    fock_coco2 = orbcs.T @ (fock0 - fockz) @ orbcs
    fock_cocv = orbos.T @ (fock0 - fockz) @ orbvs
    fock_cvcv1 = orbvs.T @ (fock0 - fockz / s) @ orbvs
    fock_cvcv2 = orbcs.T @ (fock0 + fockz / s) @ orbcs
    fock_cocv0 = orbos.T @ fockb @ orbvs  # should be specified which fock to use for cv0 block
    fock_cvov = orbos.T @ (fock0 + fockz) @ orbcs
    fock_cvcv01 = 0.5 * orbvs.T @ (focka - fockb) @ orbvs
    fock_cvcv02 = 0.5 * orbcs.T @ (focka - fockb) @ orbcs
    fock_ovov1 = orbvs.T @ (fock0 + fockz) @ orbvs
    fock_ovov2 = orbos.T @ (fock0 + fockz) @ orbos
    fock_ovcv0 = orbcs.T @ focka @ orbos
    fock_cv0cv01 = 0.5 * orbvs.T @ (focka + fockb) @ orbvs
    fock_cv0cv02 = 0.5 * orbcs.T @ (focka + fockb) @ orbcs
    fock_cooo = orbos.T @ (fock0 - fockz) @ orbcs
    fock_cvoo = orbvs.T @ fockz @ orbcs
    fock_ovoo = orbvs.T @ (fock0 + fockz) @ orbos
    fock_cv0oo = 0.5 * orbvs.T @ (focka + fockb) @ orbcs

    # diagonal part for preconditioning
    hdiag_co = (fock_coco1.diagonal()[None, :] - fock_coco2.diagonal()[:, None]).ravel()
    hdiag_cv = (fock_cvcv1.diagonal()[None, :] - fock_cvcv2.diagonal()[:, None]).ravel()
    hdiag_oo = np.array([0.0])
    hdiag_ov = (fock_ovov1.diagonal()[None, :] - fock_ovov2.diagonal()[:, None]).ravel()
    hdiag_cv0 = (fock_cv0cv01.diagonal()[None, :] - fock_cv0cv02.diagonal()[:, None]).ravel()
    hdiag = np.concatenate((hdiag_co, hdiag_cv, hdiag_oo, hdiag_ov, hdiag_cv0))

    def vind(zs):
        zs = np.asarray(zs)  # (nstates, ndim)
        zs_co = zs[:, :idx1].reshape(-1, ncs, nos)
        zs_cv = zs[:, idx1:idx2].reshape(-1, ncs, nvs)
        zs_oo = zs[:, idx2:idx3].reshape(-1, 1)
        zs_ov = zs[:, idx3:idx4].reshape(-1, nos, nvs)
        zs_cv0 = zs[:, idx4:].reshape(-1, ncs, nvs)
        dms_co = lib.einsum('xov,pv,qo->xpq', zs_co, orbos, orbcs.conj())
        dms_cv = lib.einsum('xov,pv,qo->xpq', zs_cv, orbvs, orbcs.conj())
        dms_ov = lib.einsum('xov,pv,qo->xpq', zs_ov, orbvs, orbos.conj())
        dms_cv0 = lib.einsum('xov,pv,qo->xpq', zs_cv0, orbvs, orbcs.conj())
        v1ao_co, v1ao_cv, v1ao_ov, v1ao_cv0 = vresp(dms_co, dms_cv, dms_ov, dms_cv0)
        v1mo_co = lib.einsum('xpq,qo,pv->xov', v1ao_co, orbcs, orbos.conj())
        v1mo_cv = lib.einsum('xpq,qo,pv->xov', v1ao_cv, orbcs, orbvs.conj())
        v1mo_ov = lib.einsum('xpq,qo,pv->xov', v1ao_ov, orbos, orbvs.conj())
        v1mo_cv0 = lib.einsum('xpq,qo,pv->xov', v1ao_cv0, orbcs, orbvs.conj())

        # Fock part
        v1mo_co += lib.einsum('ij,uv,xjv->xiu', np.eye(ncs), fock_coco1, zs_co)
        v1mo_co -= lib.einsum('uv,ji,xjv->xiu', np.eye(nos), fock_coco2, zs_co)
        v1mo_co += lib.einsum('ij,ub,xjb->xiu', np.eye(ncs), fock_cocv, zs_cv) * np.sqrt((s + 1) / 2 / s)
        v1mo_co -= np.einsum('ui,xv->xiu', fock_cooo, zs_oo)
        v1mo_co += lib.einsum('ij,ub,xjb->xiu', np.eye(ncs), fock_cocv0, zs_cv0) * np.sqrt(0.5)

        v1mo_cv += lib.einsum('ij,av,xjv->xia', np.eye(ncs), fock_cocv.T, zs_co) * np.sqrt((s + 1) / 2 / s)
        v1mo_cv += lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv1, zs_cv)
        v1mo_cv -= lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv2, zs_cv)
        v1mo_cv += np.einsum('ai,xv->xia', fock_cvoo, zs_oo) * np.sqrt(2 * (s + 1) / s)
        v1mo_cv -= lib.einsum('ab,vi,xvb->xia', np.eye(nvs), fock_cvov, zs_ov) * np.sqrt((s + 1) / 2 / s)
        v1mo_cv -= lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv01, zs_cv0) * np.sqrt((s + 1) / s)
        v1mo_cv += lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv02, zs_cv0) * np.sqrt((s + 1) / s)

        v1mo_ov -= lib.einsum('ab,ju,xjb->xua', np.eye(nvs), fock_cvov.T, zs_cv) * np.sqrt((s + 1) / 2 / s)
        v1mo_ov += np.einsum('au,xv->xua', fock_ovoo, zs_oo)
        v1mo_ov += lib.einsum('uv,ab,xvb->xua', np.eye(nos), fock_ovov1, zs_ov)
        v1mo_ov -= lib.einsum('ab,vu,xvb->xua', np.eye(nvs), fock_ovov2, zs_ov)
        v1mo_ov += lib.einsum('ab,ju,xjb->xua', np.eye(nvs), fock_ovcv0, zs_cv0) * np.sqrt(0.5)

        v1mo_cv0 += lib.einsum('ij,av,xjv->xia', np.eye(ncs), fock_cocv0.T, zs_co) * np.sqrt(0.5)
        v1mo_cv0 -= lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv01, zs_cv) * np.sqrt((s + 1) / s)
        v1mo_cv0 += lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv02, zs_cv) * np.sqrt((s + 1) / s)
        v1mo_cv0 -= np.einsum('ai,xv->xia', fock_cv0oo, zs_oo) * np.sqrt(2)
        v1mo_cv0 += lib.einsum('ab,vi,xvb->xia', np.eye(nvs), fock_ovcv0.T, zs_ov) * np.sqrt(0.5)
        v1mo_cv0 += lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cv0cv01, zs_cv0)
        v1mo_cv0 -= lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cv0cv02, zs_cv0)

        v1mo_oo = np.zeros((len(zs), ))
        v1mo_oo -= lib.einsum('jv,xjv->x', fock_cooo.T, zs_co)
        v1mo_oo += lib.einsum('jb,xjb->x', fock_cvoo.T, zs_cv) * np.sqrt(2 * (s + 1) / s)
        v1mo_oo += lib.einsum('vb,xvb->x', fock_ovoo.T, zs_ov)
        v1mo_oo -= lib.einsum('jb,xjb->x', fock_cv0oo.T, zs_cv0) * np.sqrt(2)

        v1mo = np.concatenate((v1mo_co.reshape(len(zs), -1),
                                v1mo_cv.reshape(len(zs), -1),
                                v1mo_oo.reshape(len(zs), -1),
                                v1mo_ov.reshape(len(zs), -1),
                                v1mo_cv0.reshape(len(zs), -1)), axis=1)
        assert v1mo.shape == zs.shape
        return v1mo
    return vind, hdiag

def gen_vind_sf(td):
    mf = td._scf
    mo_coeff = mf.mo_coeff
    assert mo_coeff[0].dtype == np.double
    mo_occ = mf.mo_occ

    mol = mf.mol
    s = (mol.nelec[0] - mol.nelec[1]) * 0.5
    assert s >= 1, 'SATDA for Sf=Si-1 only supports case that Si>=1.'

    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]
    orbcs = mo_coeff[:, csidx]
    orbos = mo_coeff[:, osidx]
    orbvs = mo_coeff[:, vsidx]
    ncs = orbcs.shape[1]
    nos = orbos.shape[1]
    nvs = orbvs.shape[1]
    nocc = ncs + nos
    nvir = nos + nvs
    idx1 = ncs * nos
    idx2 = idx1 + ncs * nvs
    idx3 = idx2 + nos * nos

    vresp, fockz = gen_rohf_response_sf(mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0, max_memory=td.max_memory)

    fock = mf.get_fock()
    focka = fock.focka
    fock0 = focka - fockz

    fock_coco0 = orbos.T @ (fock0 - fockz) @ orbos
    fock_coco1 = orbcs.T @ (fock0 + fockz) @ orbcs
    fock_coco2 = orbcs.T @ fockz @ orbcs
    fock_cocv = orbos.T @ (fock0 - fockz) @ orbvs
    fock_cooo0 = orbos.T @ (fock0 + fockz) @ orbcs
    fock_cooo1 = orbos.T @ (fock0 - fockz) @ orbcs
    fock_cvcv0 = orbvs.T @ (fock0 - fockz) @ orbvs
    fock_cvcv1 = fock_coco1
    fock_cvcv2 = orbvs.T @ fockz @ orbvs
    fock_cvcv3 = fock_coco2
    fock_cvoo = orbvs.T @ fockz @ orbcs
    fock_cvov = fock_cooo0
    fock_oooo0 = fock_coco0
    fock_oooo1 = orbos.T @ (fock0 + fockz) @ orbos
    fock_ooov0 = fock_cocv
    fock_ooov1 = orbos.T @ (fock0 + fockz) @ orbvs
    fock_ovov0 = fock_cvcv0
    fock_ovov1 = fock_oooo1
    fock_ovov2 = fock_cvcv2

    # diagonal part for preconditioning
    hdiag_co = fock_coco0.diagonal()[None, :] - fock_coco1.diagonal()[:, None]
    hdiag_co -= fock_coco2.diagonal()[:, None] * 2 / (2 * s - 1)
    hdiag_cv = fock_cvcv0.diagonal()[None, :] - fock_cvcv1.diagonal()[:, None]
    hdiag_cv -= fock_cvcv2.diagonal()[None, :] / s + fock_cvcv3.diagonal()[:, None] / s
    hdiag_oo = fock_oooo0.diagonal()[None, :] - fock_oooo1.diagonal()[:, None]
    hdiag_ov = fock_ovov0.diagonal()[None, :] - fock_ovov1.diagonal()[:, None]
    hdiag_ov -= fock_ovov2.diagonal()[None, :] * 2 / (2 * s - 1)
    hdiag = np.block([[hdiag_co, hdiag_cv], [hdiag_oo, hdiag_ov]]).ravel()

    def vind(zs):
        zs = np.asarray(zs).reshape(-1, nocc, nvir)
        zs_co = zs[:, :ncs, :nos]
        zs_cv = zs[:, :ncs, nos:]
        zs_oo = zs[:, ncs:, :nos]
        zs_ov = zs[:, ncs:, nos:]
        dms_co = lib.einsum('xov,pv,qo->xpq', zs_co, orbos, orbcs.conj())
        dms_cv = lib.einsum('xov,pv,qo->xpq', zs_cv, orbvs, orbcs.conj())
        dms_oo = lib.einsum('xov,pv,qo->xpq', zs_oo, orbos, orbos.conj())
        dms_ov = lib.einsum('xov,pv,qo->xpq', zs_ov, orbvs, orbos.conj())
        v1ao_co, v1ao_cv, v1ao_oo, v1ao_ov = vresp(dms_co, dms_cv, dms_oo, dms_ov)
        v1mo_co = lib.einsum('xpq,qo,pv->xov', v1ao_co, orbcs, orbos.conj())
        v1mo_cv = lib.einsum('xpq,qo,pv->xov', v1ao_cv, orbcs, orbvs.conj())
        v1mo_oo = lib.einsum('xpq,qo,pv->xov', v1ao_oo, orbos, orbos.conj())
        v1mo_ov = lib.einsum('xpq,qo,pv->xov', v1ao_ov, orbos, orbvs.conj())

        v1mo_co += lib.einsum('ij,uv,xjv->xiu', np.eye(ncs), fock_coco0, zs_co)
        v1mo_co -= lib.einsum('uv,ji,xjv->xiu', np.eye(nos), fock_coco1, zs_co)
        v1mo_co -= lib.einsum('uv,ji,xjv->xiu', np.eye(nos), fock_coco2, zs_co) * 2 / (2 * s - 1)
        v1mo_co += lib.einsum('ij,ub,xjb->xiu', np.eye(ncs), fock_cocv, zs_cv) * np.sqrt((2 * s + 1) / 2 / s)
        v1mo_co -= lib.einsum('uv,wi,xwv->xiu', np.eye(nos), fock_cooo0, zs_oo) * np.sqrt(2 * s / (2 * s - 1))
        v1mo_co += lib.einsum('vw,ui,xwv->xiu', np.eye(nos), fock_cooo1, zs_oo) / np.sqrt(2 * s * (2 * s - 1))

        v1mo_cv += lib.einsum('ij,av,xjv->xia', np.eye(ncs), fock_cocv.T, zs_co) * np.sqrt((2 * s + 1) / 2 / s)
        v1mo_cv += lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv0, zs_cv)
        v1mo_cv -= lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv1, zs_cv)
        v1mo_cv -= lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fock_cvcv2, zs_cv) / s
        v1mo_cv -= lib.einsum('ab,ji,xjb->xia', np.eye(nvs), fock_cvcv3, zs_cv) / s
        v1mo_cv -= lib.einsum('vw,ai,xwv->xia', np.eye(nos), fock_cvoo, zs_oo) / s * np.sqrt((2 * s + 1) / (2 * s - 1))
        v1mo_cv -= lib.einsum('ab,vi,xvb->xia', np.eye(nvs), fock_cvov, zs_ov) * np.sqrt((2 * s + 1) / 2 / s)

        v1mo_oo -= lib.einsum('vt,ju,xjv->xut', np.eye(nos), fock_cooo0.T, zs_co) * np.sqrt(2 * s / (2 * s - 1))
        v1mo_oo += lib.einsum('ut,jv,xjv->xut', np.eye(nos), fock_cooo1.T, zs_co) / np.sqrt(2 * s * (2 * s - 1))
        v1mo_oo -= lib.einsum('ut,jb,xjb->xut', np.eye(nos), fock_cvoo.T, zs_cv) / s * np.sqrt((2 * s + 1) / (2 * s - 1))
        v1mo_oo += lib.einsum('wu,tv,xwv->xut', np.eye(nos), fock_oooo0, zs_oo)
        v1mo_oo -= lib.einsum('tv,wu,xwv->xut', np.eye(nos), fock_oooo1, zs_oo)
        v1mo_oo += lib.einsum('uv,tb,xvb->xut', np.eye(nos), fock_ooov0, zs_ov) * np.sqrt(2 * s / (2 * s - 1))
        v1mo_oo -= lib.einsum('tu,vb,xvb->xut', np.eye(nos), fock_ooov1, zs_ov) / np.sqrt(2 * s * (2 * s - 1))

        v1mo_ov -= lib.einsum('ab,ju,xjb->xua', np.eye(nvs), fock_cvov.T, zs_cv) * np.sqrt((2 * s + 1) / 2 / s)
        v1mo_ov += lib.einsum('uw,av,xwv->xua', np.eye(nos), fock_ooov0.T, zs_oo) * np.sqrt(2 * s / (2 * s - 1))
        v1mo_ov -= lib.einsum('vw,au,xwv->xua', np.eye(nos), fock_ooov1.T, zs_oo) / np.sqrt(2 * s * (2 * s - 1))
        v1mo_ov += lib.einsum('uv,ab,xvb->xua', np.eye(nos), fock_ovov0, zs_ov)
        v1mo_ov -= lib.einsum('ab,vu,xvb->xua', np.eye(nvs), fock_ovov1, zs_ov)
        v1mo_ov -= lib.einsum('uv,ab,xvb->xua', np.eye(nos), fock_ovov2, zs_ov) * 2 / (2 * s - 1)

        v1mo = np.zeros_like(zs)
        v1mo[:, :ncs, :nos] = v1mo_co
        v1mo[:, :ncs, nos:] = v1mo_cv
        v1mo[:, ncs:, :nos] = v1mo_oo
        v1mo[:, ncs:, nos:] = v1mo_ov
        assert v1mo.shape == zs.shape
        return v1mo.reshape(len(v1mo), -1)
    return vind, hdiag

class SATDA(TDBase):
    '''
    Spin-Adapted TDA
    deltaS: -1 for Sf=Si-1, 0 for Sf=Si
    '''

    deltaS = getattr(__config__, 'SATDA_delta_S', -1)

    _keys = {'deltaS'}

    def init_guess(self, hdiag, nstates=None):
        if nstates is None:
            nstates = self.nstates
        n_init = min(nstates + 3, hdiag.size)
        idx = np.argsort(hdiag)[:n_init]
        x0 = np.zeros((n_init, hdiag.size))
        x0[np.arange(n_init), idx] = 1.0
        return x0

    def kernel(self, x0=None, nstates=None, deltaS=-1):
        cpu0 = (logger.process_clock(), logger.perf_counter())

        self.check_sanity()
        self.dump_flags()

        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        log = logger.Logger(self.stdout, self.verbose)

        def all_eigs(w, v, nroots, envs):
            return w, v, np.arange(w.size)

        if deltaS == 0:
            vind, hdiag = self.gen_vind_sc()
            precond = self.get_precond(hdiag)
        elif deltaS == -1:
            vind, hdiag = self.gen_vind_sf()
            precond = self.get_precond(hdiag)
            nocca, noccb = self.mol.nelec
            nvirb = self._scf.mo_occ.size - noccb
        else:
            raise ValueError('deltaS should be either 0 or -1')

        x0sym = None
        if x0 is None:
            x0 = self.init_guess(hdiag)

        self.converged, self.e, x1 = lr_eigh(
            vind,
            x0,
            precond,
            tol_residual=self.conv_tol,
            lindep=self.lindep,
            nroots=nstates,
            x0sym=x0sym,
            pick=all_eigs,
            max_cycle=self.max_cycle,
            max_memory=self.max_memory,
            verbose=log,
        )

        if deltaS == 0:
            self.xy = [(xi, 0) for xi in x1]
        elif deltaS == -1:
            self.xy = [(xi.reshape(nocca, nvirb), 0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('SATDA', *cpu0)
        self._finalize()
        return self.e, self.xy

    gen_vind_sc = gen_vind_sc
    gen_vind_sf = gen_vind_sf
