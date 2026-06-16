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
from pyscf import symm
from pyscf.data import nist

MO_BASE = getattr(__config__, 'MO_BASE', 1)
ANALYZE_THRESHOLD = 0.1

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

def _available_memory(max_memory):
    if max_memory is None:
        max_memory = 2000
    return max(2000, max_memory * 0.8 - lib.current_memory()[0])


def _eval_xc_t0(ni, xc_code, rho0a, rho0b, xctype):
    rho_t = (rho0a + rho0b) * 0.5
    fxc_t = ni.eval_xc_eff(xc_code, (rho_t, rho_t), deriv=2,
                           xctype=xctype)[2]
    if xctype == 'LDA':
        fxc_t = fxc_t[:, 0, :, 0]
        fxc_sf = (fxc_t[0, 0] + fxc_t[1, 1]
                  - fxc_t[0, 1] - fxc_t[1, 0]) * 0.25
        return fxc_sf[np.newaxis, np.newaxis]
    return (fxc_t[0, :, 0, :] + fxc_t[1, :, 1, :]
            - fxc_t[0, :, 1, :] - fxc_t[1, :, 0, :]) * 0.25


def _cache_xc_kernel_t0(mf, max_memory=None):
    mol = mf.mol
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        raise NotImplementedError(
            'Only LDA/GGA/MGGA/HF are implemented for SATDA deltaS=+1.'
        )

    ao_deriv = 0 if xctype == 'LDA' else 1
    nao = mol.nao_nr()
    dm0 = mf.to_uks().make_rdm1()
    make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1,
                                      with_lapl=False)[0]
    if max_memory is None:
        max_memory = _available_memory(mf.max_memory)

    fxc = []
    for ao, mask, weight, coords in ni.block_loop(
            mol, mf.grids, nao, ao_deriv, max_memory):
        rho0a = make_rho(0, ao, mask, xctype)
        rho0b = make_rho(1, ao, mask, xctype)
        fxc.append(_eval_xc_t0(ni, mf.xc, rho0a, rho0b, xctype))
    return np.concatenate(fxc, axis=-1)


def gen_rohf_response_splus(mf, mo_coeff=None, mo_occ=None, hermi=0,
                            max_memory=None, log=None):
    '''
    response function for Sf=Si+1 with K^SF_0 = K^t0
    '''
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mol = mf.mol
    if log is None:
        log = logger.new_logger(mf)
    assert isinstance(mf, dft.roks.ROKS) or isinstance(mf, dft.rks_symm.SymAdaptedROKS)

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    xctype = ni._xc_type(mf.xc)
    if max_memory is None:
        max_memory = mf.max_memory
    max_memory = _available_memory(max_memory)

    if mf.do_nlc():
        logger.warn(mf, "NLC contribution in gen_response is NOT included")

    if xctype in ('HF', 'NLC'):
        fxc_t0 = None
    else:
        fxc_t0 = 2.0 * _cache_xc_kernel_t0(mf, max_memory=max_memory)

    def contract_kt0(dms, hermi_):
        dms = np.asarray(dms)
        if fxc_t0 is None:
            v1ao = np.zeros_like(dms)
        else:
            time_xc = (logger.process_clock(), logger.perf_counter())
            ni_rks = dft.numint.NumInt()
            v1ao = ni_rks.nr_rks_fxc(
                mol, mf.grids, mf.xc, None, dms, 0, hermi_, None, None,
                fxc_t0, max_memory=max_memory)
            log.timer('SATDA response_splus K^T0 vref', *time_xc)

        if hybrid:
            time_jk = (logger.process_clock(), logger.perf_counter())
            vk = mf.get_k(mol, dms, hermi_) * hyb
            if omega != 0:
                vk += mf.get_k(mol, dms, hermi_, omega=omega) * (alpha - hyb)
            v1ao -= vk
            log.timer('SATDA response_splus K^T0 get_k total', *time_jk)
        return v1ao

    def vind(dms_cv):
        return contract_kt0(dms_cv, hermi)

    orbos = mo_coeff[:, np.where(mo_occ == 1)[0]]
    dmoo = orbos @ orbos.T
    delta = contract_kt0(dmoo[np.newaxis], 1)[0]
    return vind, delta

def gen_rohf_response_sc(mf, mo_coeff=None, mo_occ=None, hermi=0, max_memory=None, log=None):
    '''
    response function for Sf=Si
    '''
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mol = mf.mol
    if log is None:
        log = logger.new_logger(mf)
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

        dms0 = np.concatenate((dms_co, dms_cv, dms_ov, dms_cv0), axis=0)
        dms1 = np.concatenate((dms_co, dms_ov, dms_cv0), axis=0)

        # K^T0 part
        if xctype != 'HF':
            time_xc = (logger.process_clock(), logger.perf_counter())
            vref0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms0, 0, hermi, None, None, fxc_ref, max_memory=max_memory)
            time_xc = log.timer('SATDA response_sc K^T0 vref0', *time_xc)
            if xctype == 'LDA':
                vref1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms1, 0, hermi, None, None, fxc_ref, max_memory=max_memory)
            elif xctype =='GGA':
                vref1 = nr_rks_fxc1_gga(ni, mol, mf.grids, mf.xc, dms1, fxc_ref, max_memory=max_memory)
            elif xctype == 'MGGA':
                vref1 = nr_rks_fxc1_mgga(ni, mol, mf.grids, mf.xc, dms1, fxc_ref, max_memory=max_memory)
            log.timer('SATDA response_sc K^T0 vref1', *time_xc)
        else:
            vref0 = np.zeros_like(dms0)
            vref1 = np.zeros_like(dms1)

        if hybrid:
            time_jk = (logger.process_clock(), logger.perf_counter())
            vk = mf.get_k(mol, dms0, hermi) * hyb
            vj = mf.get_j(mol, dms1, hermi) * hyb
            if omega != 0:
                vk += mf.get_k(mol, dms0, hermi, omega=omega) * (alpha - hyb)
                vj += mf.get_j(mol, dms1, hermi, omega=omega) * (alpha - hyb)
            vref0 -= vk
            vref1 -= vj
            log.timer('SATDA response_sc K^T0 get_j/get_k total', *time_jk)

        vref0_co = vref0[:idx1]
        vref0_cv = vref0[idx1:idx2]
        vref0_ov = vref0[idx2:idx3]
        vref0_cv0 = vref0[idx3:]
        vref1_co = vref1[:idx1]
        vref1_ov = vref1[idx1:idx1+n_ov]
        vref1_cv0 = vref1[idx1+n_ov:]

        v1ao_co += vref0_co - vref1_co + np.sqrt((s + 1) / 2 / s) * vref0_cv + vref1_ov
        v1ao_cv += np.sqrt((s + 1) / 2 / s) * vref0_co + vref0_cv + np.sqrt((s + 1) / 2 / s) * vref0_ov
        v1ao_ov += vref1_co + np.sqrt((s + 1) / 2 / s) * vref0_cv - vref1_ov + vref0_ov
        v1ao_co += np.sqrt(0.5) * vref0_cv0 - np.sqrt(2.0) * vref1_cv0
        v1ao_ov += -np.sqrt(0.5) * vref0_cv0 + np.sqrt(2.0) * vref1_cv0
        v1ao_cv0 += np.sqrt(0.5) * vref0_co - np.sqrt(2.0) * vref1_co
        v1ao_cv0 += -np.sqrt(0.5) * vref0_ov + np.sqrt(2.0) * vref1_ov
        v1ao_cv0 += vref0_cv0 - 2.0 * vref1_cv0
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

def gen_rohf_response_sf(mf, mo_coeff=None, mo_occ=None, hermi=0, max_memory=None, log=None):
    '''
    response function for Sf=Si-1
    '''
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mol = mf.mol
    if log is None:
        log = logger.new_logger(mf)
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
            time_xc = (logger.process_clock(), logger.perf_counter())
            vref0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms0, 0, hermi, None, None, fxc_ref, max_memory=max_memory)
            time_xc = log.timer('SATDA response_sf vref0', *time_xc)
            if xctype == 'LDA':
                vref1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms1, 0, hermi, None, None, fxc_ref, max_memory=max_memory)
            elif xctype =='GGA':
                vref1 = nr_rks_fxc1_gga(ni, mol, mf.grids, mf.xc, dms1, fxc_ref, max_memory=max_memory)
            elif xctype == 'MGGA':
                vref1 = nr_rks_fxc1_mgga(ni, mol, mf.grids, mf.xc, dms1, fxc_ref, max_memory=max_memory)
            log.timer('SATDA response_sf vref1', *time_xc)
        else:
            vref0 = np.zeros_like(dms0)
            vref1 = np.zeros_like(dms1)

        # HF part
        if hybrid:
            time_jk = (logger.process_clock(), logger.perf_counter())
            vk = mf.get_k(mol, dms0, hermi) * hyb
            vj = mf.get_j(mol, dms1, hermi) * hyb
            if omega != 0:
                vk += mf.get_k(mol, dms0, hermi, omega=omega) * (alpha - hyb)
                vj += mf.get_j(mol, dms1, hermi, omega=omega) * (alpha - hyb)
            vref0 -= vk
            vref1 -= vj
            log.timer('SATDA response_sf get_j/get_k total', *time_jk)
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

    log = logger.new_logger(td)
    vresp, fockz = gen_rohf_response_sc(mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0,
                                        max_memory=td.max_memory, log=log)

    fock = mf.get_fock()
    focka = fock.focka
    fockb = focka - 2.0 * fockz
    fock0 = focka - fockz

    fock_coco1 = orbos.T @ (fock0 - fockz) @ orbos
    fock_coco2 = orbcs.T @ (fock0 - fockz) @ orbcs
    fock_cocv = orbos.T @ (fock0 - fockz) @ orbvs
    fock_cvcv1 = orbvs.T @ (fock0 - fockz / s) @ orbvs
    fock_cvcv2 = orbcs.T @ (fock0 + fockz / s) @ orbcs
    fock_cocv0 = orbos.T @ fockb @ orbvs
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
        time0 = time1 = (logger.process_clock(), logger.perf_counter())
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
        time1 = log.timer('SATDA gen_vind_sc make density matrices', *time1)
        v1ao_co, v1ao_cv, v1ao_ov, v1ao_cv0 = vresp(dms_co, dms_cv, dms_ov, dms_cv0)
        time1 = log.timer('SATDA gen_vind_sc response vind total', *time1)
        v1mo_co = lib.einsum('xpq,qo,pv->xov', v1ao_co, orbcs, orbos.conj())
        v1mo_cv = lib.einsum('xpq,qo,pv->xov', v1ao_cv, orbcs, orbvs.conj())
        v1mo_ov = lib.einsum('xpq,qo,pv->xov', v1ao_ov, orbos, orbvs.conj())
        v1mo_cv0 = lib.einsum('xpq,qo,pv->xov', v1ao_cv0, orbcs, orbvs.conj())
        time1 = log.timer('SATDA gen_vind_sc AO->MO transform', *time1)

        # Fock part TODO: accelerate
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
        time1 = log.timer('SATDA gen_vind_sc Fock part', *time1)

        v1mo = np.concatenate((v1mo_co.reshape(len(zs), -1),
                                v1mo_cv.reshape(len(zs), -1),
                                v1mo_oo.reshape(len(zs), -1),
                                v1mo_ov.reshape(len(zs), -1),
                                v1mo_cv0.reshape(len(zs), -1)), axis=1)
        assert v1mo.shape == zs.shape
        time1 = log.timer('SATDA gen_vind_sc pack result', *time1)
        log.timer('SATDA gen_vind_sc total', *time0)
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

    log = logger.new_logger(td)
    vresp, fockz = gen_rohf_response_sf(mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0,
                                        max_memory=td.max_memory, log=log)

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
        time0 = time1 = (logger.process_clock(), logger.perf_counter())
        zs = np.asarray(zs).reshape(-1, nocc, nvir)
        zs_co = zs[:, :ncs, :nos]
        zs_cv = zs[:, :ncs, nos:]
        zs_oo = zs[:, ncs:, :nos]
        zs_ov = zs[:, ncs:, nos:]
        dms_co = lib.einsum('xov,pv,qo->xpq', zs_co, orbos, orbcs.conj())
        dms_cv = lib.einsum('xov,pv,qo->xpq', zs_cv, orbvs, orbcs.conj())
        dms_oo = lib.einsum('xov,pv,qo->xpq', zs_oo, orbos, orbos.conj())
        dms_ov = lib.einsum('xov,pv,qo->xpq', zs_ov, orbvs, orbos.conj())
        time1 = log.timer('SATDA gen_vind_sf make density matrices', *time1)
        v1ao_co, v1ao_cv, v1ao_oo, v1ao_ov = vresp(dms_co, dms_cv, dms_oo, dms_ov)
        time1 = log.timer('SATDA gen_vind_sf response vind total', *time1)
        v1mo_co = lib.einsum('xpq,qo,pv->xov', v1ao_co, orbcs, orbos.conj())
        v1mo_cv = lib.einsum('xpq,qo,pv->xov', v1ao_cv, orbcs, orbvs.conj())
        v1mo_oo = lib.einsum('xpq,qo,pv->xov', v1ao_oo, orbos, orbos.conj())
        v1mo_ov = lib.einsum('xpq,qo,pv->xov', v1ao_ov, orbos, orbvs.conj())
        time1 = log.timer('SATDA gen_vind_sf AO->MO transform', *time1)

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
        time1 = log.timer('SATDA gen_vind_sf Fock part', *time1)

        v1mo = np.zeros_like(zs)
        v1mo[:, :ncs, :nos] = v1mo_co
        v1mo[:, :ncs, nos:] = v1mo_cv
        v1mo[:, ncs:, :nos] = v1mo_oo
        v1mo[:, ncs:, nos:] = v1mo_ov
        assert v1mo.shape == zs.shape
        time1 = log.timer('SATDA gen_vind_sf pack result', *time1)
        log.timer('SATDA gen_vind_sf total', *time0)
        return v1mo.reshape(len(v1mo), -1)
    return vind, hdiag

def gen_vind_splus(td):
    mf = td._scf
    mo_coeff = mf.mo_coeff
    assert mo_coeff[0].dtype == np.double
    mo_occ = mf.mo_occ

    csidx = np.where(mo_occ == 2)[0]
    vsidx = np.where(mo_occ == 0)[0]
    orbcs = mo_coeff[:, csidx]
    orbvs = mo_coeff[:, vsidx]
    ncs = orbcs.shape[1]
    nvs = orbvs.shape[1]

    log = logger.new_logger(td)
    vresp, delta = gen_rohf_response_splus(mf, mo_coeff=mo_coeff,
                                           mo_occ=mo_occ, hermi=0,
                                           max_memory=td.max_memory, log=log)

    fock = mf.get_fock()
    focka = fock.focka
    fockb = focka - delta

    fock_cvcv = orbvs.T @ focka @ orbvs
    fock_coco = orbcs.T @ fockb @ orbcs

    hdiag = (fock_cvcv.diagonal()[None, :]
             - fock_coco.diagonal()[:, None]).ravel()

    def vind(zs):
        time0 = time1 = (logger.process_clock(), logger.perf_counter())
        zs = np.asarray(zs).reshape(-1, ncs, nvs)
        dms_cv = lib.einsum('xia,pa,qi->xpq', zs, orbvs, orbcs.conj())
        time1 = log.timer('SATDA gen_vind_splus make density matrices', *time1)
        v1ao_cv = vresp(dms_cv)
        time1 = log.timer('SATDA gen_vind_splus response vind total', *time1)
        v1mo_cv = lib.einsum('xpq,qi,pa->xia', v1ao_cv,
                             orbcs, orbvs.conj())
        time1 = log.timer('SATDA gen_vind_splus AO->MO transform', *time1)

        v1mo_cv += lib.einsum('ab,xib->xia', fock_cvcv, zs)
        v1mo_cv -= lib.einsum('ji,xja->xia', fock_coco, zs)
        time1 = log.timer('SATDA gen_vind_splus Fock part', *time1)

        v1mo = v1mo_cv.reshape(len(zs), -1)
        time1 = log.timer('SATDA gen_vind_splus pack result', *time1)
        log.timer('SATDA gen_vind_splus total', *time0)
        return v1mo
    return vind, hdiag

class SATDA(TDBase):
    '''
    Spin-Adapted TDA
    deltaS: -1 for Sf=Si-1, 0 for Sf=Si, 1 for Sf=Si+1
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

    def kernel(self, x0=None, nstates=None):
        cpu0 = (logger.process_clock(), logger.perf_counter())

        self.check_sanity()
        self.dump_flags()

        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        if self.deltaS == -1:
            nstates += 1
        log = logger.Logger(self.stdout, self.verbose)

        def all_eigs(w, v, nroots, envs):
            return w, v, np.arange(w.size)

        if self.deltaS == 0:
            vind, hdiag = self.gen_vind_sc()
            precond = self.get_precond(hdiag)
        elif self.deltaS == -1:
            vind, hdiag = self.gen_vind_sf()
            precond = self.get_precond(hdiag)
            csidx, osidx, vsidx = _satda_orbital_indices(self)
            nocc = len(csidx) + len(osidx)
            nvir = len(osidx) + len(vsidx)
        elif self.deltaS == 1:
            vind, hdiag = self.gen_vind_splus()
            precond = self.get_precond(hdiag)
            csidx, _, vsidx = _satda_orbital_indices(self)
            ncs = len(csidx)
            nvs = len(vsidx)
        else:
            raise ValueError('deltaS should be -1, 0, or 1')

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

        if self.deltaS == 0:
            self.xy = [(xi, 0) for xi in x1]
        elif self.deltaS == -1:
            self.xy = [(xi.reshape(nocc, nvir), 0) for xi in x1]
            mask = abs(self.e) > 1e-8
            self.e = self.e[mask]
            self.xy = [xy for xy, keep in zip(self.xy, mask) if keep]
            self.nstates = len(self.e)
        elif self.deltaS == 1:
            self.xy = [(xi.reshape(ncs, nvs), 0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('SATDA', *cpu0)
        self._finalize()
        return self.e, self.xy

    gen_vind_sc = gen_vind_sc
    gen_vind_sf = gen_vind_sf
    gen_vind_splus = gen_vind_splus


def _analyze_wfnsym(tdobj, x_sym, x):
    possible_sym = np.asarray(x_sym)[np.abs(np.asarray(x)) > ANALYZE_THRESHOLD]
    wfnsym = symm.MULTI_IRREPS
    ids = possible_sym[possible_sym != symm.MULTI_IRREPS]
    if len(ids) > 0 and np.all(ids == ids[0]):
        wfnsym = int(ids[0])
    if wfnsym == symm.MULTI_IRREPS:
        return wfnsym, '???'
    return wfnsym, _safe_irrep_id2name(tdobj.mol.groupname, wfnsym)


def _safe_irrep_id2name(groupname, irrep_id):
    try:
        return symm.irrep_id2name(groupname, int(irrep_id))
    except KeyError:
        return '???'


def _satda_block_slices(nc, no, nv, deltaS):
    co = nc * no
    cv = nc * nv
    oo = no * no if deltaS == -1 else 1
    ov = no * nv
    p_co = 0
    p_cv = p_co + co
    p_oo = p_cv + cv
    p_ov = p_oo + oo
    p_cv0 = p_ov + ov
    blocks = {
        'CO(1)': slice(p_co, p_cv),
        'CV(1)': slice(p_cv, p_oo),
        'OO(1)': slice(p_oo, p_ov),
        'OV(1)': slice(p_ov, p_cv0),
    }
    if deltaS == 0:
        blocks['CV(0)'] = slice(p_cv0, p_cv0 + cv)
    return blocks


def _satda_orbital_indices(tdobj):
    mo_occ = np.asarray(tdobj._scf.mo_occ)
    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]
    return csidx, osidx, vsidx


def _log_state(log, tdobj, istate, e_ev, wfnsymid=None, wfnsymlabel=None):
    mol = tdobj.mol
    mf = tdobj._scf
    if wfnsymid is None:
        log.note('Excited State %3d: %12.5f eV', istate + 1, e_ev)
        return
    refsym = mf.get_wfnsym()
    if isinstance(refsym, str):
        refsym = symm.irrep_name2id(mol.groupname, refsym)
    else:
        refsym = int(np.asarray(refsym).ravel()[0])
    if refsym == symm.MULTI_IRREPS or wfnsymid == symm.MULTI_IRREPS:
        statesymlabel = '???'
    else:
        statesymid = symm.direct_prod(
            np.array([int(wfnsymid)]), np.array([refsym]), mol.groupname,
        ).ravel()[0]
        if statesymid == symm.MULTI_IRREPS:
            statesymlabel = '???'
        else:
            statesymlabel = _safe_irrep_id2name(mol.groupname, statesymid)
    log.note(
        'Excited State %3d: %4s (State: %4s) %12.5f eV',
        istate + 1, wfnsymlabel, statesymlabel, e_ev,
    )


def _analyze_sc(tdobj, verbose=None):
    log = logger.new_logger(tdobj, verbose)
    mol = tdobj.mol
    mf = tdobj._scf
    csidx, osidx, vsidx = _satda_orbital_indices(tdobj)
    nc = len(csidx)
    no = len(osidx)
    nv = len(vsidx)
    slices = _satda_block_slices(nc, no, nv, deltaS=0)

    if mol.symmetry and mol.groupname != 'C1':
        orbsym = mf.get_orbsym(mf.mo_coeff)
        x_syms = {
            'CO(1)': symm.direct_prod(orbsym[csidx], orbsym[osidx], mol.groupname),
            'CV(1)': symm.direct_prod(orbsym[csidx], orbsym[vsidx], mol.groupname),
            'OV(1)': symm.direct_prod(orbsym[osidx], orbsym[vsidx], mol.groupname),
            'CV(0)': symm.direct_prod(orbsym[csidx], orbsym[vsidx], mol.groupname),
        }
    else:
        x_syms = None

    for i in range(tdobj.nstates):
        x, y = tdobj.xy[i]
        x = np.asarray(x).reshape(-1)
        e_ev = np.asarray(tdobj.e[i]) * nist.HARTREE2EV

        if x_syms is None:
            _log_state(log, tdobj, i, e_ev)
        else:
            syms = []
            vals = []
            for key in ('CO(1)', 'CV(1)', 'OV(1)', 'CV(0)'):
                x_block = x[slices[key]]
                vals.append(np.max(np.abs(x_block), initial=0.0))
                syms.append((key, x_block.reshape(x_syms[key].shape)))
            key, x_block = syms[int(np.argmax(vals))]
            wfnsymid, wfnsymlabel = _analyze_wfnsym(tdobj, x_syms[key], x_block)
            _log_state(log, tdobj, i, e_ev, wfnsymid, wfnsymlabel)

        if log.verbose >= logger.INFO:
            x_co1 = x[slices['CO(1)']].reshape(nc, no)
            x_cv1 = x[slices['CV(1)']].reshape(nc, nv)
            x_oo1 = x[slices['OO(1)']]
            x_ov1 = x[slices['OV(1)']].reshape(no, nv)
            x_cv0 = x[slices['CV(0)']].reshape(nc, nv)
            for c, o in zip(*np.where(np.abs(x_co1) > ANALYZE_THRESHOLD)):
                log.info('    CO(1) %4d -> %4d %12.5f', csidx[c] + MO_BASE, osidx[o] + MO_BASE, x_co1[c, o])
            for c, v in zip(*np.where(np.abs(x_cv1) > ANALYZE_THRESHOLD)):
                log.info('    CV(1) %4d -> %4d %12.5f', csidx[c] + MO_BASE, vsidx[v] + MO_BASE, x_cv1[c, v])
            if abs(x_oo1[0]) > ANALYZE_THRESHOLD:
                log.info('    OO(1) %12.5f', x_oo1[0])
            for o, v in zip(*np.where(np.abs(x_ov1) > ANALYZE_THRESHOLD)):
                log.info('    OV(1) %4d -> %4d %12.5f', osidx[o] + MO_BASE, vsidx[v] + MO_BASE, x_ov1[o, v])
            for c, v in zip(*np.where(np.abs(x_cv0) > ANALYZE_THRESHOLD)):
                log.info('    CV(0) %4d -> %4d %12.5f', csidx[c] + MO_BASE, vsidx[v] + MO_BASE, x_cv0[c, v])
    return tdobj


def _analyze_sf(tdobj, verbose=None):
    log = logger.new_logger(tdobj, verbose)
    mol = tdobj.mol
    mf = tdobj._scf
    csidx, osidx, vsidx = _satda_orbital_indices(tdobj)
    nc = len(csidx)
    no = len(osidx)
    nv = len(vsidx)
    nocc = nc + no
    nvir = no + nv
    # slices = _satda_block_slices(nc, no, nv, deltaS=-1)

    if mol.symmetry and mol.groupname != 'C1':
        orbsym = mf.get_orbsym(mf.mo_coeff)
        x_syms = {
            'CO(1)': symm.direct_prod(orbsym[csidx], orbsym[osidx], mol.groupname),
            'CV(1)': symm.direct_prod(orbsym[csidx], orbsym[vsidx], mol.groupname),
            'OO(1)': symm.direct_prod(orbsym[osidx], orbsym[osidx], mol.groupname),
            'OV(1)': symm.direct_prod(orbsym[osidx], orbsym[vsidx], mol.groupname),
        }
    else:
        x_syms = None

    for i in range(tdobj.nstates):
        x, y = tdobj.xy[i]
        x = np.asarray(x).reshape(nocc, nvir)
        e_ev = np.asarray(tdobj.e[i]) * nist.HARTREE2EV

        x_co1 = x[:nc, :no]
        x_cv1 = x[:nc, no:]
        x_oo1 = x[nc:, :no]
        x_ov1 = x[nc:, no:]

        if x_syms is None:
            _log_state(log, tdobj, i, e_ev)
        else:
            x_blocks = {
                'CO(1)': x_co1,
                'CV(1)': x_cv1,
                'OO(1)': x_oo1,
                'OV(1)': x_ov1,
            }
            key = max(x_blocks, key=lambda k: np.max(np.abs(x_blocks[k]), initial=0.0))
            wfnsymid, wfnsymlabel = _analyze_wfnsym(tdobj, x_syms[key], x_blocks[key])
            _log_state(log, tdobj, i, e_ev, wfnsymid, wfnsymlabel)

        if log.verbose >= logger.INFO:
            for c, o in zip(*np.where(np.abs(x_co1) > ANALYZE_THRESHOLD)):
                log.info('    CO(1) %4d -> %4d %12.5f', csidx[c] + MO_BASE, osidx[o] + MO_BASE, x_co1[c, o])
            for c, v in zip(*np.where(np.abs(x_cv1) > ANALYZE_THRESHOLD)):
                log.info('    CV(1) %4d -> %4d %12.5f', csidx[c] + MO_BASE, vsidx[v] + MO_BASE, x_cv1[c, v])
            for o1, o2 in zip(*np.where(np.abs(x_oo1) > ANALYZE_THRESHOLD)):
                log.info('    OO(1) %4d -> %4d %12.5f', osidx[o1] + MO_BASE, osidx[o2] + MO_BASE, x_oo1[o1, o2])
            for o, v in zip(*np.where(np.abs(x_ov1) > ANALYZE_THRESHOLD)):
                log.info('    OV(1) %4d -> %4d %12.5f', osidx[o] + MO_BASE, vsidx[v] + MO_BASE, x_ov1[o, v])
    return tdobj


def _analyze_splus(tdobj, verbose=None):
    log = logger.new_logger(tdobj, verbose)
    mol = tdobj.mol
    mf = tdobj._scf
    csidx, osidx, vsidx = _satda_orbital_indices(tdobj)
    nc = len(csidx)
    nv = len(vsidx)

    if mol.symmetry and mol.groupname != 'C1':
        orbsym = mf.get_orbsym(mf.mo_coeff)
        x_sym = symm.direct_prod(orbsym[csidx], orbsym[vsidx],
                                 mol.groupname)
    else:
        x_sym = None

    nstates = min(tdobj.nstates, len(tdobj.xy))
    for i in range(nstates):
        x, y = tdobj.xy[i]
        x_cv = np.asarray(x).reshape(nc, nv)
        e_ev = np.asarray(tdobj.e[i]) * nist.HARTREE2EV

        if x_sym is None:
            _log_state(log, tdobj, i, e_ev)
        else:
            wfnsymid, wfnsymlabel = _analyze_wfnsym(tdobj, x_sym, x_cv)
            _log_state(log, tdobj, i, e_ev, wfnsymid, wfnsymlabel)

        if log.verbose >= logger.INFO:
            for c, v in zip(*np.where(np.abs(x_cv) > ANALYZE_THRESHOLD)):
                log.info('    CV(+1) %4d -> %4d %12.5f',
                         csidx[c] + MO_BASE, vsidx[v] + MO_BASE, x_cv[c, v])
    return tdobj


def analyze(tdobj, verbose=None):
    if tdobj.deltaS == 0:
        return _analyze_sc(tdobj, verbose)
    if tdobj.deltaS == -1:
        return _analyze_sf(tdobj, verbose)
    if tdobj.deltaS == 1:
        return _analyze_splus(tdobj, verbose)
    raise ValueError('deltaS should be -1, 0, or 1')


SATDA.analyze = analyze


def transition_dipole(tdobj, ref=1, state=None):
    """
    Transition dipole moments between SATDA DeltaS = -1 excited states.

    Parameters
    ----------
    tdobj : SATDA object
        Need tdobj.deltaS == -1 and tdobj.xy.
    ref : int
        1-based reference excited-state index.
    state : int or array-like or None
        1-based target excited-state index/indices. If None, all states except ref.

    Returns
    -------
    pol : ndarray, shape (nstates, 3)
        <ref| r |state> in length gauge.
    """
    mf = tdobj._scf
    mol = mf.mol

    deltaS = getattr(tdobj, "deltaS", getattr(tdobj, "DeltaS", None))
    assert deltaS == -1

    s = (mol.nelec[0] - mol.nelec[1]) * 0.5
    assert s >= 1

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    assert mo_occ.ndim == 1

    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]

    ncs = len(csidx)
    nos = len(osidx)
    nvs = len(vsidx)
    nocc = ncs + nos
    nvir = nos + nvs
    nmo = mo_coeff.shape[1]

    if state is None:
        states = np.arange(tdobj.nstates) + 1
    else:
        states = np.atleast_1d(state).astype(int)

    states = states[states != ref]
    ref0 = ref - 1
    states0 = states - 1

    def get_x(i):
        x = tdobj.xy[i]
        if isinstance(x, (tuple, list)):
            x = x[0]
        return np.asarray(x).reshape(nocc, nvir)

    mx = get_x(ref0)
    nxs = np.asarray([get_x(i) for i in states0])

    # x_co: j v, x_cv: j b, x_oo: w v, x_ov: v b
    m_co = mx[:ncs, :nos].conj()
    m_cv = mx[:ncs, nos:].conj()
    m_oo = mx[ncs:, :nos].conj()
    m_ov = mx[ncs:, nos:].conj()

    n_co = nxs[:, :ncs, :nos]
    n_cv = nxs[:, :ncs, nos:]
    n_oo = nxs[:, ncs:, :nos]
    n_ov = nxs[:, ncs:, nos:]

    nstate = len(states0)
    gamma = np.zeros((nstate, nmo, nmo), dtype=np.result_type(mx, nxs, complex))
    ist = np.arange(nstate)

    def add(rows, cols, block):
        gamma[np.ix_(ist, rows, cols)] += block

    a = np.sqrt(2 * s / (2 * s - 1))
    b = 1 / np.sqrt(2 * s * (2 * s - 1))
    # c = np.sqrt((2 * s - 1) / (2 * s))
    f = np.sqrt((2 * s + 1) / (2 * s))

    tr_moo = np.einsum("tt->", m_oo)
    tr_noo = np.einsum("ntt->n", n_oo)

    # OO-OO
    add(osidx, osidx,  lib.einsum("ut,nuv->ntv", m_oo, n_oo))
    add(osidx, osidx, -lib.einsum("ut,nwt->nwu", m_oo, n_oo))

    # CO-CO
    add(osidx, osidx,  lib.einsum("iu,niv->nuv", m_co, n_co))
    add(csidx, csidx, -lib.einsum("iu,nju->nji", m_co, n_co))

    # CV-CV
    add(csidx, csidx, -lib.einsum("ia,nja->nji", m_cv, n_cv))
    add(vsidx, vsidx,  lib.einsum("ia,nib->nab", m_cv, n_cv))

    # OV-OV
    add(osidx, osidx, -lib.einsum("ua,nva->nvu", m_ov, n_ov))
    add(vsidx, vsidx,  lib.einsum("ua,nub->nab", m_ov, n_ov))

    # OO-CO and CO-OO
    add(csidx, osidx,
        -a * lib.einsum("ut,njt->nju", m_oo, n_co)
        + b * tr_moo * n_co)

    add(osidx, csidx,
        -a * lib.einsum("iu,nwu->nwi", m_co, n_oo)
        + b * lib.einsum("iu,n->nui", m_co, tr_noo))

    # OO-OV and OV-OO
    add(osidx, vsidx,
         a * lib.einsum("ut,nub->ntb", m_oo, n_ov)
        - b * tr_moo * n_ov)

    add(vsidx, osidx,
         a * lib.einsum("ua,nuv->nav", m_ov, n_oo)
        - b * lib.einsum("ua,n->nau", m_ov, tr_noo))

    # CO-CV and CV-CO
    add(osidx, vsidx, f * lib.einsum("iu,nib->nub", m_co, n_cv))
    add(vsidx, osidx, f * lib.einsum("ia,niv->nav", m_cv, n_co))

    # OV-CV and CV-OV
    add(csidx, osidx, -f * lib.einsum("ua,nja->nju", m_ov, n_cv))
    add(osidx, csidx, -f * lib.einsum("ia,nva->nvi", m_cv, n_ov))

    dip_ao = mol.intor_symmetric("int1e_r", comp=3)
    dip_mo = lib.einsum("up,xuv,vq->xpq", mo_coeff.conj(), dip_ao, mo_coeff)

    pol = lib.einsum("npq,xpq->nx", gamma, dip_mo)
    return pol.real

def oscillator_strength(tdobj, ref=1, state=None):
    if state is None:
        states = np.arange(tdobj.nstates) + 1
    else:
        states = np.atleast_1d(state)
    states = states[states != ref]

    trans_dip = transition_dipole(tdobj, ref, states)

    ref -= 1
    states -= 1
    es = tdobj.e[states] - tdobj.e[ref]
    f = (2./3.) * lib.einsum('n,nx,nx->n', es, trans_dip.conj(), trans_dip).real
    if isinstance(state, int):
        return f[0]
    else:
        return f

SATDA.transition_dipole = transition_dipole
SATDA.oscillator_strength = oscillator_strength
