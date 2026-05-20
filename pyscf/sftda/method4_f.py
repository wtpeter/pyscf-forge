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
# Spin-Adapted Spin-Flip-Down TDA with Multicollinear Functionals
#
# Author: Tai Wang <wtpeter@pku.edu.cn>
# Ref:
# Mol. Phys. 2026, e2631735
#

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf import dft
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.tdscf.uhf import TDBase
from pyscf import __config__

# from pyscf.sftda.scf_genrep_sftd import gen_uhf_response_sf
# from pyscf.sftda.sasf import spin_square
# from pyscf.sftda.sasf import analyze
from pyscf.dft.gen_grid import NBINS
from pyscf.tdscf._lr_eig import eigh as lr_eigh

def get_a_sasf(mf, mo_energy=None, mo_coeff=None, mo_occ=None, collinear_samples=20):
    """
    A_[i,a,j,b]for A_abab
    """
    # Standard spin-flip TDA
    #assert isinstance(mf, dft.roks.ROKS)
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mol = mf.mol
    nao, nmo = mo_coeff.shape
    si = (mol.nelec[0] - mol.nelec[1]) * 0.5

    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]
    orbcs = mo_coeff[:, csidx]
    orbos = mo_coeff[:, osidx]
    orbvs = mo_coeff[:, vsidx]
    ncs = orbcs.shape[1]
    nos = orbos.shape[1]
    nvs = orbvs.shape[1]
    orbo = np.concatenate((orbcs, orbos), axis=1)
    orbv = np.concatenate((orbos, orbvs), axis=1)
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]

    fock = mf.get_fock()
    focka = fock.focka

    hybrid = False
    hyb = 1
    omega = 0
    alpha = 0
    xctype = 'HF'
    make_rho = None
    max_memory = None

    if isinstance(mf, dft.KohnShamDFT):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        dm0 = mf.to_uks().make_rdm1()
        make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]
        xctype = ni._xc_type(mf.xc)
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory * 0.8 - mem_now)

        def eval_xc_t0(rho0a, rho0b):
            rho_t = (rho0a + rho0b) * 0.5
            fxc_t = ni.eval_xc_eff(
                mf.xc, (rho_t, rho_t), deriv=2, xctype=xctype
            )[2]
            if xctype == 'LDA':
                fxc_t = fxc_t[:, 0, :, 0]
                fxc_sf = (
                    fxc_t[0, 0] + fxc_t[1, 1]
                    - fxc_t[0, 1] - fxc_t[1, 0]
                ) * 0.25
                return fxc_sf[np.newaxis, np.newaxis]
            return (
                fxc_t[0, :, 0, :] + fxc_t[1, :, 1, :]
                - fxc_t[0, :, 1, :] - fxc_t[1, :, 0, :]
            ) * 0.25

        def build_open_shell_xc_kt0_ao():
            v_open = np.zeros_like(focka)
            if nos == 0 or xctype in ('HF', 'NLC'):
                return v_open

            if xctype == 'LDA':
                ao_deriv = 0
                for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                    rho0a = make_rho(0, ao, mask, xctype)
                    rho0b = make_rho(1, ao, mask, xctype)
                    fxc_sf = eval_xc_t0(rho0a, rho0b)
                    rho_os = lib.einsum('rp,pi->ri', ao, orbos)
                    rho_oo = np.einsum('ru,ru->r', rho_os, rho_os)
                    w_open = 2 * fxc_sf[0, 0] * weight * rho_oo
                    v_open += lib.einsum('rp,rq,r->pq', ao, ao, w_open)
            elif xctype in ('GGA', 'MGGA'):
                ao_deriv = 1
                for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                    rho0a = make_rho(0, ao, mask, xctype)
                    rho0b = make_rho(1, ao, mask, xctype)
                    fxc_sf = eval_xc_t0(rho0a, rho0b)
                    wfxc = 2 * fxc_sf * weight
                    rho_os = lib.einsum('xrp,pi->xri', ao, orbos)
                    rho_oo = np.empty((4, ao.shape[1]))
                    rho_oo[0] = np.einsum('ru,ru->r', rho_os[0], rho_os[0])
                    rho_oo[1:4] = 2 * np.einsum('xru,ru->xr', rho_os[1:4], rho_os[0])
                    if xctype == 'MGGA':
                        tau_oo = np.einsum('xru,xru->r', rho_os[1:4], rho_os[1:4]) * 0.5
                        rho_oo = np.vstack([rho_oo, tau_oo[np.newaxis]])

                    w_open = np.einsum('xyr,xr->yr', wfxc, rho_oo)
                    ao0 = ao[0]
                    v_open += lib.einsum('rp,rq,r->pq', ao0, ao0, w_open[0])
                    for x in range(1, 4):
                        v_open += lib.einsum('rp,rq,r->pq', ao[x], ao0, w_open[x])
                        v_open += lib.einsum('rp,rq,r->pq', ao0, ao[x], w_open[x])
                    if xctype == 'MGGA':
                        for x in range(1, 4):
                            v_open += 0.5 * lib.einsum('rp,rq,r->pq', ao[x], ao[x], w_open[4])
            return v_open

        def build_open_shell_exact_exchange_ao():
            if nos == 0 or not hybrid:
                return np.zeros_like(focka)

            dm_open = orbos @ orbos.T
            k_open = mf.get_k(mol, dm_open, hermi=1) * hyb
            if omega != 0:
                k_fac = alpha - hyb
                if k_fac != 0:
                    k_open += mf.get_k(mol, dm_open, hermi=1, omega=omega) * k_fac
            return k_open

        fockb = focka - build_open_shell_xc_kt0_ao() + build_open_shell_exact_exchange_ao()
    else:
        fockb = focka.copy()

    a = np.zeros((nocc, nvir, nocc, nvir))

    fockv = orbv.T @ fockb @ orbv
    focko = orbo.T @ focka @ orbo
    a += np.einsum('ij,ab->iajb', np.eye(nocc), fockv)
    a -= np.einsum('ab,ij->iajb', np.eye(nvir), focko)

    def add_hf_(a, hyb=1):
        eri = ao2mo.general(mol, [orbo, orbo, orbv, orbv], compact=False)
        eri = eri.reshape(nocc, nocc, nvir, nvir)
        a -= np.einsum('ijba->iajb', eri) * hyb

    if isinstance(mf, dft.KohnShamDFT):
        if hybrid:
            add_hf_(a, hyb)
            if omega != 0:
                with mol.with_range_coulomb(omega):
                    eri = ao2mo.general(mol, [orbo, orbo, orbv, orbv], compact=False)
                    eri = eri.reshape(nocc, nocc, nvir, nvir)
                    k_fac = alpha - hyb
                    a -= np.einsum('ijba->iajb', eri) * k_fac

        if collinear_samples >= 0:
            if xctype == 'LDA':
                ao_deriv = 0
                for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                    rho0a = make_rho(0, ao, mask, xctype)
                    rho0b = make_rho(1, ao, mask, xctype)
                    fxc_sf = eval_xc_t0(rho0a, rho0b)
                    wfxc = fxc_sf[0, 0] * weight
                    rho_o = lib.einsum('rp,pi->ri', ao, orbo)
                    rho_v = lib.einsum('rp,pi->ri', ao, orbv)
                    rho_ov = np.einsum('ri,ra->ria', rho_o, rho_v)
                    w_ov = np.einsum('ria,r->ria', rho_ov, wfxc)
                    iajb = lib.einsum('ria,rjb->iajb', rho_ov, w_ov) * 2  # 2 for f_xx + f_yy
                    a += iajb

            elif xctype == 'GGA':
                ao_deriv = 1
                for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                    rho0a = make_rho(0, ao, mask, xctype)
                    rho0b = make_rho(1, ao, mask, xctype)
                    fxc_sf = eval_xc_t0(rho0a, rho0b)
                    wfxc = fxc_sf * weight
                    rho_o = lib.einsum('xrp,pi->xri', ao, orbo)
                    rho_v = lib.einsum('xrp,pi->xri', ao, orbv)
                    rho_ov = np.einsum('xri,ra->xria', rho_o, rho_v[0])
                    rho_ov[1:4] += np.einsum('ri,xra->xria', rho_o[0], rho_v[1:4])
                    w_ov = np.einsum('xyr,xria->yria', wfxc, rho_ov)
                    iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov) * 2
                    a += iajb

            elif xctype == 'HF':
                pass

            elif xctype == 'NLC':
                pass

            elif xctype == 'MGGA':
                ao_deriv = 1
                for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                    rho0a = make_rho(0, ao, mask, xctype)
                    rho0b = make_rho(1, ao, mask, xctype)
                    fxc_sf = eval_xc_t0(rho0a, rho0b)
                    wfxc = fxc_sf * weight
                    rho_o = lib.einsum('xrp,pi->xri', ao, orbo)
                    rho_v = lib.einsum('xrp,pi->xri', ao, orbv)
                    rho_ov = np.einsum('xri,ra->xria', rho_o, rho_v[0])
                    rho_ov[1:4] += np.einsum('ri,xra->xria', rho_o[0], rho_v[1:4])
                    tau_ov = np.einsum('xri,xra->ria', rho_o[1:4], rho_v[1:4]) * 0.5
                    rho_ov = np.vstack([rho_ov, tau_ov[np.newaxis]])
                    w_ov = np.einsum('xyr,xria->yria', wfxc, rho_ov)
                    iajb = lib.einsum('xria,xrjb->iajb', w_ov, rho_ov) * 2
                    a += iajb

            if mf.do_nlc():
                raise NotImplementedError(
                    'vv10 nlc not implemented in get_ab(). '
                    'However the nlc contribution is small in TDDFT, '
                    'so feel free to take the risk and comment out this line.'
                )
    else:
        add_hf_(a, hyb=1)

    # Correction of spin adaptation
    a_cvcv = np.zeros((ncs, nvs, ncs, nvs))
    a_coco = np.zeros((ncs, nos, ncs, nos))
    a_ovov = np.zeros((nos, nvs, nos, nvs))
    a_cvco = np.zeros((ncs, nvs, ncs, nos))
    a_cvov = np.zeros((ncs, nvs, nos, nvs))
    a_coov = np.zeros((ncs, nos, nos, nvs))
    a_oooo = np.zeros((nos, nos, nos, nos))
    a_cvoo = np.zeros((ncs, nvs, nos, nos))
    a_cooo = np.zeros((ncs, nos, nos, nos))
    a_ovoo = np.zeros((nos, nvs, nos, nos))

    focks = 0.5 * (fockb - focka)
    focksc = orbcs.T @ focks @ orbcs
    focksv = orbvs.T @ focks @ orbvs
    fockbvo = orbvs.T @ fockb @ orbos
    fockaoc = orbos.T @ focka @ orbcs
    fockscv = orbcs.T @ focks @ orbvs
    fockbco = orbcs.T @ fockb @ orbos
    fockavo = orbvs.T @ focka @ orbos
    a_cvcv += (lib.einsum('ij,ab->iajb', np.eye(ncs), focksv) + lib.einsum('ab,ji->iajb', np.eye(nvs), focksc)) / si
    a_coco += lib.einsum('uv,ji->iujv', np.eye(nos), focksc) * 2 / (2 * si - 1)
    a_ovov += lib.einsum('uv,ab->uavb', np.eye(nos), focksv) * 2 / (2 * si - 1)
    a_cvco += lib.einsum('ij,av->iajv', np.eye(ncs), fockbvo) * (np.sqrt((2 * si + 1) / (2 * si)) - 1)
    a_cvov -= lib.einsum('ab,vi->iavb', np.eye(nvs), fockaoc) * (np.sqrt((2 * si + 1) / (2 * si)) - 1)
    a_cvoo += lib.einsum('vw,ia->iawv', np.eye(nos), fockscv) * np.sqrt((2 * si + 1) / (2 * si - 1)) / si
    a_cooo += lib.einsum('vw,iu->iuwv', np.eye(nos), fockbco) / np.sqrt(2 * si * (2 * si - 1))
    a_cooo -= lib.einsum('uv,iw->iuwv', np.eye(nos), fockaoc.T) * (np.sqrt(2 * si / (2 * si - 1)) - 1)
    a_ovoo -= lib.einsum('vw,au->uawv', np.eye(nos), fockavo) / np.sqrt(2 * si * (2 * si - 1))
    a_ovoo += lib.einsum('wu,av->uawv', np.eye(nos), fockbvo) * (np.sqrt(2 * si / (2 * si - 1)) - 1)

    if isinstance(mf, dft.KohnShamDFT):
        pass
    else:
        hyb = 1
        omega = 0

    if hybrid:
        eri_coco = ao2mo.general(mol, [orbos, orbcs, orbcs, orbos], compact=False).reshape(nos, ncs, ncs, nos)
        a_coco -= np.einsum('uijv->iujv', eri_coco) * hyb / (2 * si - 1)
        eri_ovov = ao2mo.general(mol, [orbvs, orbos, orbos, orbvs], compact=False).reshape(nvs, nos, nos, nvs)
        a_ovov -= np.einsum('auvb->uavb', eri_ovov) * hyb / (2 * si - 1)
        eri_cvco = ao2mo.general(mol, [orbvs, orbos, orbcs, orbcs], compact=False).reshape(nvs, nos, ncs, ncs)
        a_cvco -= np.einsum('avji->iajv', eri_cvco) * hyb * (np.sqrt((2 * si + 1) / (2 * si)) - 1)
        eri_cvov = ao2mo.general(mol, [orbvs, orbvs, orbos, orbcs], compact=False).reshape(nvs, nvs, nos, ncs)
        a_cvov -= np.einsum('abvi->iavb', eri_cvov) * hyb * (np.sqrt((2 * si + 1) / (2 * si)) - 1)
        eri_coov1 = ao2mo.general(mol, [orbos, orbcs, orbos, orbvs], compact=False).reshape(nos, ncs, nos, nvs)
        eri_coov2 = ao2mo.general(mol, [orbos, orbvs, orbos, orbcs], compact=False).reshape(nos, nvs, nos, ncs)
        a_coov += np.einsum('uivb->iuvb', eri_coov1) * hyb / (2 * si - 1)
        a_coov -= np.einsum('ubvi->iuvb', eri_coov2) * hyb / (2 * si - 1)
        eri_cvoo = ao2mo.general(mol, [orbvs, orbos, orbos, orbcs], compact=False).reshape(nvs, nos, nos, ncs)
        a_cvoo -= np.einsum('avwi->iawv', eri_cvoo) * hyb * (np.sqrt((2 * si + 1) / (2 * si - 1)) - 1)
        eri_cooo = ao2mo.general(mol, [orbos, orbos, orbos, orbcs], compact=False).reshape(nos, nos, nos, ncs)
        a_cooo -= np.einsum('uvwi->iuwv', eri_cooo) * hyb * (np.sqrt(2 * si / (2 * si - 1)) - 1)
        eri_ovoo = ao2mo.general(mol, [orbvs, orbos, orbos, orbos], compact=False).reshape(nvs, nos, nos, nos)
        a_ovoo -= np.einsum('avwu->uawv', eri_ovoo) * hyb * (np.sqrt(2 * si / (2 * si - 1)) - 1)
        if omega != 0:
            with mol.with_range_coulomb(omega):
                k_fac = alpha - hyb
                eri_coco_rsh = ao2mo.general(mol, [orbos, orbcs, orbcs, orbos], compact=False).reshape(
                    nos, ncs, ncs, nos
                )
                a_coco -= np.einsum('uijv->iujv', eri_coco_rsh) * k_fac / (2 * si - 1)
                eri_ovov_rsh = ao2mo.general(mol, [orbvs, orbos, orbos, orbvs], compact=False).reshape(
                    nvs, nos, nos, nvs
                )
                a_ovov -= np.einsum('auvb->uavb', eri_ovov_rsh) * k_fac / (2 * si - 1)
                eri_cvco_rsh = ao2mo.general(mol, [orbvs, orbos, orbcs, orbcs], compact=False).reshape(
                    nvs, nos, ncs, ncs
                )
                a_cvco -= np.einsum('avji->iajv', eri_cvco_rsh) * k_fac * (np.sqrt((2 * si + 1) / (2 * si)) - 1)
                eri_cvov_rsh = ao2mo.general(mol, [orbvs, orbvs, orbos, orbcs], compact=False).reshape(
                    nvs, nvs, nos, ncs
                )
                a_cvov -= np.einsum('abvi->iavb', eri_cvov_rsh) * k_fac * (np.sqrt((2 * si + 1) / (2 * si)) - 1)
                eri_coov1_rsh = ao2mo.general(mol, [orbos, orbcs, orbos, orbvs], compact=False).reshape(
                    nos, ncs, nos, nvs
                )
                eri_coov2_rsh = ao2mo.general(mol, [orbos, orbvs, orbos, orbcs], compact=False).reshape(
                    nos, nvs, nos, ncs
                )
                a_coov += np.einsum('uivb->iuvb', eri_coov1_rsh) * k_fac / (2 * si - 1)
                a_coov -= np.einsum('ubvi->iuvb', eri_coov2_rsh) * k_fac / (2 * si - 1)
                eri_cvoo_rsh = ao2mo.general(mol, [orbvs, orbos, orbos, orbcs], compact=False).reshape(
                    nvs, nos, nos, ncs
                )
                a_cvoo -= np.einsum('avwi->iawv', eri_cvoo_rsh) * k_fac * (np.sqrt((2 * si + 1) / (2 * si - 1)) - 1)
                eri_cooo_rsh = ao2mo.general(mol, [orbos, orbos, orbos, orbcs], compact=False).reshape(
                    nos, nos, nos, ncs
                )
                a_cooo -= np.einsum('uvwi->iuwv', eri_cooo_rsh) * k_fac * (np.sqrt(2 * si / (2 * si - 1)) - 1)
                eri_ovoo_rsh = ao2mo.general(mol, [orbvs, orbos, orbos, orbos], compact=False).reshape(
                    nvs, nos, nos, nos
                )
                a_ovoo -= np.einsum('avwu->uawv', eri_ovoo_rsh) * k_fac * (np.sqrt(2 * si / (2 * si - 1)) - 1)

    if collinear_samples >= 0:
        k_coco = np.zeros((ncs, nos, ncs, nos))
        k_ovov = np.zeros((nos, nvs, nos, nvs))
        k_cvco = np.zeros((ncs, nvs, ncs, nos))
        k_cvov = np.zeros((ncs, nvs, nos, nvs))
        k_coov = np.zeros((ncs, nos, nos, nvs))
        k_cvoo = np.zeros((ncs, nvs, nos, nos))
        k_cooo = np.zeros((ncs, nos, nos, nos))
        k_ovoo = np.zeros((nos, nvs, nos, nos))

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                fxc_sf = eval_xc_t0(rho0a, rho0b)
                wfxc = 2 * fxc_sf[0, 0] * weight
                rho_cs = lib.einsum('rp,pi->ri', ao, orbcs)
                rho_os = lib.einsum('rp,pi->ri', ao, orbos)
                rho_vs = lib.einsum('rp,pi->ri', ao, orbvs)
                rho_cc = np.einsum('ri,ra->ria', rho_cs, rho_cs)
                rho_co = np.einsum('ri,ra->ria', rho_cs, rho_os)
                rho_cv = np.einsum('ri,ra->ria', rho_cs, rho_vs)
                rho_oo = np.einsum('ri,ra->ria', rho_os, rho_os)
                rho_ov = np.einsum('ri,ra->ria', rho_os, rho_vs)
                rho_vv = np.einsum('ri,ra->ria', rho_vs, rho_vs)
                w_co = np.einsum('ria,r->ria', rho_co, wfxc)
                w_oo = np.einsum('ria,r->ria', rho_oo, wfxc)
                w_ov = np.einsum('ria,r->ria', rho_ov, wfxc)
                k_coco += lib.einsum('rij,ruv->iujv', rho_cc, w_oo) / (2 * si - 1)
                k_ovov += lib.einsum('rab,ruv->uavb', rho_vv, w_oo) / (2 * si - 1)
                k_cvco += lib.einsum('ria,rjv->iajv', rho_cv, w_co) * (np.sqrt((2 * si + 1) / (2 * si)) - 1)
                k_cvov += lib.einsum('ria,rvb->iavb', rho_cv, w_ov) * (np.sqrt((2 * si + 1) / (2 * si)) - 1)
                k_coov -= lib.einsum('rub,riv->iuvb', rho_ov, w_co) / (2 * si - 1)
                k_coov += lib.einsum('riu,rvb->iuvb', rho_co, w_ov) / (2 * si - 1)
                k_cvoo += lib.einsum('ria,rvw->iawv', rho_cv, w_oo) * (np.sqrt((2 * si + 1) / (2 * si - 1)) - 1)
                k_cooo += lib.einsum('riu,rvw->iuwv', rho_co, w_oo) * (np.sqrt(2 * si / (2 * si - 1)) - 1)
                k_ovoo += lib.einsum('rua,rvw->uawv', rho_ov, w_oo) * (np.sqrt(2 * si / (2 * si - 1)) - 1)
        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                fxc_sf = eval_xc_t0(rho0a, rho0b)
                wfxc = 2 * fxc_sf * weight  # 注意这里跟LDA一样，乘上了2

                rho_cs = lib.einsum('xrp,pi->xri', ao, orbcs)
                rho_os = lib.einsum('xrp,pi->xri', ao, orbos)
                rho_vs = lib.einsum('xrp,pi->xri', ao, orbvs)

                def make_pair_gga(r1, r2):
                    r12 = np.einsum('xri,rj->xrij', r1, r2[0])
                    r12[1:4] += np.einsum('ri,xrj->xrij', r1[0], r2[1:4])
                    return r12

                rho_cc = make_pair_gga(rho_cs, rho_cs)
                rho_co = make_pair_gga(rho_cs, rho_os)
                rho_cv = make_pair_gga(rho_cs, rho_vs)
                rho_oo = make_pair_gga(rho_os, rho_os)
                rho_ov = make_pair_gga(rho_os, rho_vs)
                rho_vv = make_pair_gga(rho_vs, rho_vs)

                w_co = np.einsum('xyr,xria->yria', wfxc, rho_co)
                w_oo = np.einsum('xyr,xria->yria', wfxc, rho_oo)
                w_ov = np.einsum('xyr,xria->yria', wfxc, rho_ov)

                k_coco += lib.einsum('xrij,xruv->iujv', rho_cc, w_oo) / (2 * si - 1)
                k_ovov += lib.einsum('xrab,xruv->uavb', rho_vv, w_oo) / (2 * si - 1)
                k_cvco += lib.einsum('xria,xrjv->iajv', rho_cv, w_co) * (np.sqrt((2 * si + 1) / (2 * si)) - 1)
                k_cvov += lib.einsum('xria,xrvb->iavb', rho_cv, w_ov) * (np.sqrt((2 * si + 1) / (2 * si)) - 1)
                k_coov -= lib.einsum('xrub,xriv->iuvb', rho_ov, w_co) / (2 * si - 1)
                k_coov += lib.einsum('xriu,xrvb->iuvb', rho_co, w_ov) / (2 * si - 1)
                k_cvoo += lib.einsum('xria,xrvw->iawv', rho_cv, w_oo) * (np.sqrt((2 * si + 1) / (2 * si - 1)) - 1)
                k_cooo += lib.einsum('xriu,xrvw->iuwv', rho_co, w_oo) * (np.sqrt(2 * si / (2 * si - 1)) - 1)
                k_ovoo += lib.einsum('xrua,xrvw->uawv', rho_ov, w_oo) * (np.sqrt(2 * si / (2 * si - 1)) - 1)

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                fxc_sf = eval_xc_t0(rho0a, rho0b)
                wfxc = 2 * fxc_sf * weight

                rho_cs = lib.einsum('xrp,pi->xri', ao, orbcs)
                rho_os = lib.einsum('xrp,pi->xri', ao, orbos)
                rho_vs = lib.einsum('xrp,pi->xri', ao, orbvs)

                def make_pair_mgga(r1, r2):
                    r12 = np.einsum('xri,rj->xrij', r1, r2[0])
                    r12[1:4] += np.einsum('ri,xrj->xrij', r1[0], r2[1:4])
                    tau12 = np.einsum('xri,xrj->rij', r1[1:4], r2[1:4]) * 0.5
                    return np.vstack([r12, tau12[np.newaxis]])

                rho_cc = make_pair_mgga(rho_cs, rho_cs)
                rho_co = make_pair_mgga(rho_cs, rho_os)
                rho_cv = make_pair_mgga(rho_cs, rho_vs)
                rho_oo = make_pair_mgga(rho_os, rho_os)
                rho_ov = make_pair_mgga(rho_os, rho_vs)
                rho_vv = make_pair_mgga(rho_vs, rho_vs)

                w_co = np.einsum('xyr,xria->yria', wfxc, rho_co)
                w_oo = np.einsum('xyr,xria->yria', wfxc, rho_oo)
                w_ov = np.einsum('xyr,xria->yria', wfxc, rho_ov)

                k_coco += lib.einsum('xrij,xruv->iujv', rho_cc, w_oo) / (2 * si - 1)
                k_ovov += lib.einsum('xrab,xruv->uavb', rho_vv, w_oo) / (2 * si - 1)
                k_cvco += lib.einsum('xria,xrjv->iajv', rho_cv, w_co) * (np.sqrt((2 * si + 1) / (2 * si)) - 1)
                k_cvov += lib.einsum('xria,xrvb->iavb', rho_cv, w_ov) * (np.sqrt((2 * si + 1) / (2 * si)) - 1)
                k_coov -= lib.einsum('xrub,xriv->iuvb', rho_ov, w_co) / (2 * si - 1)
                k_coov += lib.einsum('xriu,xrvb->iuvb', rho_co, w_ov) / (2 * si - 1)
                k_cvoo += lib.einsum('xria,xrvw->iawv', rho_cv, w_oo) * (np.sqrt((2 * si + 1) / (2 * si - 1)) - 1)
                k_cooo += lib.einsum('xriu,xrvw->iuwv', rho_co, w_oo) * (np.sqrt(2 * si / (2 * si - 1)) - 1)
                k_ovoo += lib.einsum('xrua,xrvw->uawv', rho_ov, w_oo) * (np.sqrt(2 * si / (2 * si - 1)) - 1)

        elif xctype == 'HF' or xctype == 'NLC':
            pass

        a_coco += k_coco
        a_ovov += k_ovov
        a_cvco += k_cvco
        a_cvov += k_cvov
        a_coov += k_coov
        a_cvoo += k_cvoo
        a_cooo += k_cooo
        a_ovoo += k_ovoo

    a_cocv = a_cvco.transpose(2, 3, 0, 1)
    a_ovcv = a_cvov.transpose(2, 3, 0, 1)
    a_ovco = a_coov.transpose(2, 3, 0, 1)
    a_oocv = a_cvoo.transpose(2, 3, 0, 1)
    a_ooco = a_cooo.transpose(2, 3, 0, 1)
    a_ooov = a_ovoo.transpose(2, 3, 0, 1)

    delta_a = np.zeros_like(a)
    delta_a[:ncs, :nos, :ncs, :nos] = a_coco
    delta_a[:ncs, :nos, :ncs, nos:] = a_cocv
    delta_a[:ncs, :nos, ncs:, :nos] = a_cooo
    delta_a[:ncs, :nos, ncs:, nos:] = a_coov
    delta_a[:ncs, nos:, :ncs, :nos] = a_cvco
    delta_a[:ncs, nos:, :ncs, nos:] = a_cvcv
    delta_a[:ncs, nos:, ncs:, :nos] = a_cvoo
    delta_a[:ncs, nos:, ncs:, nos:] = a_cvov
    delta_a[ncs:, :nos, :ncs, :nos] = a_ooco
    delta_a[ncs:, :nos, :ncs, nos:] = a_oocv
    delta_a[ncs:, :nos, ncs:, :nos] = a_oooo
    delta_a[ncs:, :nos, ncs:, nos:] = a_ooov
    delta_a[ncs:, nos:, :ncs, :nos] = a_ovco
    delta_a[ncs:, nos:, :ncs, nos:] = a_ovcv
    delta_a[ncs:, nos:, ncs:, :nos] = a_ovoo
    delta_a[ncs:, nos:, ncs:, nos:] = a_ovov
    a += delta_a

    return a


def _zero_excitation_vector(mo_occ):
    nmo = mo_occ.size
    nocc_a = np.count_nonzero(mo_occ > 0)
    nocc_b = np.count_nonzero(mo_occ == 2)
    nvir_b = nmo - nocc_b
    nopen = nocc_a - nocc_b

    if nopen <= 0:
        return np.zeros((nocc_a, nvir_b))

    v_red = np.zeros((nocc_a, nvir_b))
    factor = 1.0 / np.sqrt(nopen)
    for u in range(nopen):
        v_red[nocc_b + u, u] = factor
    return v_red


class TDA_SASF(TDBase):
    collinear_samples = getattr(__config__, 'tdscf_uhf_sf_SF-TDA_collinear_samples', 20)
    remove = False
    _keys = {'collinear_samples', 'remove', 'extype'}

    def __init__(self, mf, collinear_samples=20, remove=False):
        TDBase.__init__(self, mf)
        self.collinear_samples = collinear_samples
        self.remove = remove
        self.extype = getattr(self, 'extype', 1)
        if self.extype != 1:
            raise NotImplementedError("Only spin flip down is allowed!")

    def get_a_sasf(self, mf=None, collinear_samples=None):
        if mf is None:
            mf = self._scf
        if collinear_samples is None:
            collinear_samples = self.collinear_samples
        return get_a_sasf(mf, collinear_samples=collinear_samples)

    def kernel(self, x0=None, nstates=None, extype=None):
        """
        Full diagonalization solver for the Method-1 SASF-TDA matrix.
        """
        cpu0 = (logger.process_clock(), logger.perf_counter())

        self.check_sanity()
        self.dump_flags()

        if extype is not None and extype != 1:
            raise NotImplementedError("Only spin flip down is allowed!")

        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)

        a = self.get_a_sasf()
        nocc, nvir = a.shape[:2]
        nov = nocc * nvir
        amat = a.reshape(nov, nov)

        if self.remove:
            v_red = _zero_excitation_vector(self._scf.mo_occ).reshape(-1)
            norm_vred = np.linalg.norm(v_red)
            if norm_vred < 1e-12:
                raise RuntimeError('Cannot remove zero-excitation component: no open-shell orbitals found.')
            v_red = v_red / norm_vred

            log.note('Remove redundant GS component: True')
            pspace = scipy.linalg.null_space(v_red.reshape(1, -1))
            amat_proj = pspace.conj().T @ amat @ pspace
            e, x_proj = scipy.linalg.eig(amat_proj)
            e = e.real
            x = (pspace @ x_proj).real

            sort_idx = np.argsort(e)
            e = e[sort_idx]
            x = x[:, sort_idx]

            e_vred = float((v_red.conj() @ amat @ v_red).real)
            insert_pos = int(np.searchsorted(e, e_vred))
            e = np.insert(e, insert_pos, e_vred)
            x = np.insert(x, insert_pos, v_red, axis=1)
        else:
            e, x = scipy.linalg.eig(amat)
            e = e.real
            x = x.real

            sort_idx = np.argsort(e)
            e = e[sort_idx]
            x = x[:, sort_idx]

        keep = np.arange(min(nstates, e.size))
        if keep.size < nstates:
            log.warn('Requested %d states, but the full Method-1 SASF-TDA '
                     'matrix only provides %d states.', nstates, keep.size)

        self.e = e[keep]
        self.xy = []
        for state_idx in keep:
            x_matrix = x[:, state_idx].reshape(nocc, nvir)
            norm = np.linalg.norm(x_matrix)
            if norm > 1e-12:
                x_matrix = x_matrix / norm
            self.xy.append((x_matrix, 0))
        self.converged = np.ones(len(self.e), dtype=bool)

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('SASF-TDA full diagonalization', *cpu0)
        self._finalize()
        return self.e, self.xy



TDA = TDA_SASF
