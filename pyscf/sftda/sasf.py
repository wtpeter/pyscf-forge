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
from pyscf import lib
from pyscf import gto
from pyscf import dft
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.tdscf.uhf import TDBase
from pyscf import __config__

# from pyscf.sftda.scf_genrep_sftd import gen_uhf_response_sf
from pyscf.sftda.numint2c_sftd import mcfun_eval_xc_adapter_sf
from pyscf.sftda.numint2c_sftd import cache_xc_kernel_sf
from pyscf.sftda.uhf_sf import TDA_SF
from pyscf.dft.gen_grid import NBINS
from pyscf.dft.numint import _scale_ao_sparse, _dot_ao_ao_sparse, _dot_ao_dm_sparse, _contract_rho_sparse
from pyscf.tdscf._lr_eig import eigh as lr_eigh


def get_a_sasf(mf, mo_energy=None, mo_coeff=None, mo_occ=None, collinear_samples=20):
    """
    A_[i,a,j,b]for A_abab
    """
    # Standard spin-flip TDA
    assert isinstance(mf, dft.roks.ROKS)
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
    fockb = fock.fockb

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
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        if hybrid:
            add_hf_(a, hyb)
            if omega != 0:
                with mol.with_range_coulomb(omega):
                    eri = ao2mo.general(mol, [orbo, orbo, orbv, orbv], compact=False)
                    eri = eri.reshape(nocc, nocc, nvir, nvir)
                    k_fac = alpha - hyb
                    a -= np.einsum('ijba->iajb', eri) * k_fac

        if collinear_samples >= 0:
            dm0 = mf.to_uks().make_rdm1()
            make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]
            xctype = ni._xc_type(mf.xc)
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory * 0.8 - mem_now)

            # create a mc object to use mcfun.
            nimc = dft.numint2c.NumInt2C()
            nimc.collinear = 'mcol'
            nimc.collinear_samples = collinear_samples
            eval_xc_eff = mcfun_eval_xc_adapter_sf(nimc, mf.xc)

            if xctype == 'LDA':
                ao_deriv = 0
                for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                    rho0a = make_rho(0, ao, mask, xctype)
                    rho0b = make_rho(1, ao, mask, xctype)
                    rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                    fxc_sf = eval_xc_eff(mf.xc, rho_z, deriv=2, xctype=xctype)[2]
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
                    rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                    fxc_sf = eval_xc_eff(mf.xc, rho_z, deriv=2, xctype=xctype)[2]
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
                    rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                    fxc_sf = eval_xc_eff(mf.xc, rho_z, deriv=2, xctype=xctype)[2]
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

    focks = 0.5 * (fock.fockb - fock.focka)
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
                rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                fxc_sf = eval_xc_eff(mf.xc, rho_z, deriv=2, xctype=xctype)[2]
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
                rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                fxc_sf = eval_xc_eff(mf.xc, rho_z, deriv=2, xctype=xctype)[2]
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
                rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                fxc_sf = eval_xc_eff(mf.xc, rho_z, deriv=2, xctype=xctype)[2]
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


def gen_uhf_response_sf1(mf, mo_coeff=None, mo_occ=None, hermi=0, collinear_samples=20, max_memory=None):
    """
    Generate a function to compute the response function with type
    K_{pr,qs} x_{sr}
    """
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    mol = mf.mol

    if isinstance(mf, dft.KohnShamDFT):
        ni = dft.numint2c.NumInt2C()
        ni.collinear = 'mcol'
        ni.collinear_samples = collinear_samples
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

        if mf.nlc or ni.libxc.is_nlc(mf.xc):
            logger.warn(
                mf,
                'NLC functional found in DFT object. Its contribution is not included in the TDDFT response function.',
            )
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        if collinear_samples >= 0:
            fxc = 2.0 * cache_xc_kernel_sf(ni, mol, mf.grids, mf.xc, mo_coeff, mo_occ, deriv=2, spin=1)[2]

        if max_memory is None:
            max_memory = mf.max_memory
        max_memory = max_memory * 0.8 - lib.current_memory()[0]

        def vind(dm1):
            if collinear_samples < 0:
                v1 = np.zeros_like(dm1)
            # TODO: Add pure funtional part.
            else:
                in2 = dft.numint.NumInt()
                xctype = in2._xc_type(mf.xc)
                if xctype == 'LDA':
                    dm0 = None
                    v1 = in2.nr_rks_fxc(
                        mol, mf.grids, mf.xc, dm0, dm1, 0, hermi, None, None, fxc, max_memory=max_memory
                    )
                elif xctype == 'GGA':
                    v1 = nr_rks_fxc1_gga(in2, mol, mf.grids, mf.xc, dm1, fxc, max_memory=max_memory)
                elif xctype == 'MGGA':
                    v1 = nr_rks_fxc1_mgga(in2, mol, mf.grids, mf.xc, dm1, fxc, max_memory=max_memory)
            if hybrid:
                vj = mf.get_j(mol, dm1) * hyb
                if omega != 0:
                    k_fac = alpha - hyb
                    vj += mf.get_j(mol, dm1, omega=omega) * k_fac
                v1 -= vj
            return v1

        return vind
    else:  # HF

        def vind(dm1):
            vj = mf.get_j(mol, dm1)
            return -vj

        return vind


def gen_uhf_response_sf_merged(mf, mo_coeff=None, mo_occ=None, hermi=0, collinear_samples=200, max_memory=None):
    """
    Merged response function generator that computes BOTH:
    1. Standard response (K_{pr,qs} x_{qs}) for all 4 blocks (co, cv, oo, ov)
    2. Exchange-like response (K_{pr,qs} x_{sr}) for 2 blocks (co, ov)
    It evaluates fxc ONLY ONCE to massively boost performance.
    """
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    mol = mf.mol

    is_dft = isinstance(mf, dft.KohnShamDFT)

    if is_dft:
        ni = dft.numint2c.NumInt2C()
        ni.collinear = 'mcol'
        ni.collinear_samples = collinear_samples
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

        if mf.nlc or ni.libxc.is_nlc(mf.xc):
            lib.logger.warn(
                mf,
                'NLC functional found in DFT object. Its contribution is not included in the TDDFT response function.',
            )
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        if collinear_samples >= 0:
            fxc = 2.0 * cache_xc_kernel_sf(ni, mol, mf.grids, mf.xc, mo_coeff, mo_occ, deriv=2, spin=1)[2]
        else:
            fxc = None
    else:
        hybrid = False

    if max_memory is None:
        max_memory = mf.max_memory
    max_memory = max_memory * 0.8 - lib.current_memory()[0]

    def vind(dms_co, dms_cv, dms_oo, dms_ov):
        dms0 = np.concatenate((dms_co, dms_cv, dms_oo, dms_ov), axis=0)
        dms1 = np.concatenate((dms_co, dms_ov), axis=0)

        if is_dft:
            if collinear_samples < 0:
                v1ao0 = np.zeros_like(dms0)
                v1ao1 = np.zeros_like(dms1)
            else:
                in2 = dft.numint.NumInt()
                xctype = in2._xc_type(mf.xc)
                v1ao0 = in2.nr_rks_fxc(
                    mol, mf.grids, mf.xc, None, dms0, 0, hermi, None, None, fxc, max_memory=max_memory
                )

                if xctype == 'LDA':
                    v1ao1 = in2.nr_rks_fxc(
                        mol, mf.grids, mf.xc, None, dms1, 0, hermi, None, None, fxc, max_memory=max_memory
                    )
                elif xctype == 'GGA':
                    v1ao1 = nr_rks_fxc1_gga(in2, mol, mf.grids, mf.xc, dms1, fxc, max_memory=max_memory)
                elif xctype == 'MGGA':
                    v1ao1 = nr_rks_fxc1_mgga(in2, mol, mf.grids, mf.xc, dms1, fxc, max_memory=max_memory)

            if hybrid:
                if omega == 0:
                    vk = mf.get_k(mol, dms0, hermi) * hyb
                elif alpha == 0:
                    vk = mf.get_k(mol, dms0, hermi, omega=-omega) * hyb
                elif hyb == 0:
                    vk = mf.get_k(mol, dms0, hermi, omega=omega) * alpha
                else:
                    vk = mf.get_k(mol, dms0, hermi) * hyb
                    vk += mf.get_k(mol, dms0, hermi, omega=omega) * (alpha - hyb)
                v1ao0 -= vk

                vj = mf.get_j(mol, dms1) * hyb
                if omega != 0:
                    k_fac = alpha - hyb
                    vj += mf.get_j(mol, dms1, omega=omega) * k_fac
                v1ao1 -= vj

        else:  # HF
            vk = mf.get_k(mol, dms0, hermi)
            v1ao0 = -vk
            vj = mf.get_j(mol, dms1)
            v1ao1 = -vj

        n_co = len(dms_co)
        n_cv = len(dms_cv)
        n_oo = len(dms_oo)

        idx1 = n_co
        idx2 = idx1 + n_cv
        idx3 = idx2 + n_oo

        v1ao_co0 = v1ao0[:idx1]
        v1ao_cv0 = v1ao0[idx1:idx2]
        v1ao_oo0 = v1ao0[idx2:idx3]
        v1ao_ov0 = v1ao0[idx3:]

        v1ao_co1 = v1ao1[:n_co]
        v1ao_ov1 = v1ao1[n_co:]

        return v1ao_co0, v1ao_cv0, v1ao_oo0, v1ao_ov0, v1ao_co1, v1ao_ov1

    return vind


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


class TDA_SASF(TDBase):
    collinear_samples = getattr(__config__, 'tdscf_uhf_sf_SF-TDA_collinear_samples', 20)
    _keys = {'collinear_samples', 'sftda'}

    def __init__(self, mf, collinear_samples=20):
        TDBase.__init__(self, mf)
        self.collinear_samples = collinear_samples
        if isinstance(self._scf, dft.KohnShamDFT):
            umf = self._scf.to_uks()
        else:
            umf = self._scf.to_uhf()
        self.sftda = TDA_SF(umf, collinear_samples=collinear_samples, extype=1)

    def gen_vind(self):
        mf = self._scf
        mo_coeff = mf.mo_coeff
        assert mo_coeff[0].dtype == np.double
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
        fockb = fock.fockb
        focks = 0.5 * (fockb - focka)

        focka_cc = orbcs.conj().T @ focka @ orbcs
        focka_oo = orbos.conj().T @ focka @ orbos
        focka_oc = orbos.conj().T @ focka @ orbcs
        focka_ov = orbos.conj().T @ focka @ orbvs
        fockb_oo = orbos.conj().T @ fockb @ orbos
        fockb_oc = orbos.conj().T @ fockb @ orbcs
        fockb_ov = orbos.conj().T @ fockb @ orbvs
        fockb_vv = orbvs.conj().T @ fockb @ orbvs
        focks_cc = orbcs.conj().T @ focks @ orbcs
        focks_vv = orbvs.conj().T @ focks @ orbvs
        focks_vc = orbvs.conj().T @ focks @ orbcs

        hdiag_co = fockb_oo.diagonal()[None, :] - focka_cc.diagonal()[:, None]
        hdiag_co += focks_cc.diagonal()[:, None] * 2 / (2 * si - 1)
        hdiag_cv = fockb_vv.diagonal()[None, :] - focka_cc.diagonal()[:, None]
        hdiag_cv += focks_vv.diagonal()[None, :] / si + focks_cc.diagonal()[:, None] / si
        hdiag_oo = fockb_oo.diagonal()[None, :] - focka_oo.diagonal()[:, None]
        hdiag_ov = fockb_vv.diagonal()[None, :] - focka_oo.diagonal()[:, None]
        hdiag_ov += focks_vv.diagonal()[None, :] * 2 / (2 * si - 1)
        hdiag = np.block([[hdiag_co, hdiag_cv], [hdiag_oo, hdiag_ov]]).ravel()

        # vresp0 = gen_uhf_response_sf(umf, hermi=0, collinear_samples=self.collinear_samples)
        # vresp1 = gen_uhf_response_sf1(umf, hermi=0, collinear_samples=self.collinear_samples)
        vresp_merged = gen_uhf_response_sf_merged(self.sftda._scf, hermi=0, collinear_samples=self.collinear_samples)

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
            # dms0 = np.concatenate((dms_co, dms_cv, dms_oo, dms_ov), axis=0)
            # dms1 = np.concatenate((dms_co, dms_ov), axis=0)
            # v1ao_co0, v1ao_cv0, v1ao_oo0, v1ao_ov0 = vresp0(dms0).reshape(4, -1, nao, nao)
            # v1ao_co1, v1ao_ov1 = vresp1(dms1).reshape(2, -1, nao, nao)
            v1ao_co0, v1ao_cv0, v1ao_oo0, v1ao_ov0, v1ao_co1, v1ao_ov1 = vresp_merged(dms_co, dms_cv, dms_oo, dms_ov)

            v1mo_coco0 = lib.einsum('xpq,qo,pv->xov', v1ao_co0, orbcs, orbos.conj())
            v1mo_cvco0 = lib.einsum('xpq,qo,pv->xov', v1ao_co0, orbcs, orbvs.conj())
            v1mo_ooco0 = lib.einsum('xpq,qo,pv->xov', v1ao_co0, orbos, orbos.conj())
            v1mo_ovco0 = lib.einsum('xpq,qo,pv->xov', v1ao_co0, orbos, orbvs.conj())

            v1mo_cocv0 = lib.einsum('xpq,qo,pv->xov', v1ao_cv0, orbcs, orbos.conj())
            v1mo_cvcv0 = lib.einsum('xpq,qo,pv->xov', v1ao_cv0, orbcs, orbvs.conj())
            v1mo_oocv0 = lib.einsum('xpq,qo,pv->xov', v1ao_cv0, orbos, orbos.conj())
            v1mo_ovcv0 = lib.einsum('xpq,qo,pv->xov', v1ao_cv0, orbos, orbvs.conj())

            v1mo_cooo0 = lib.einsum('xpq,qo,pv->xov', v1ao_oo0, orbcs, orbos.conj())
            v1mo_cvoo0 = lib.einsum('xpq,qo,pv->xov', v1ao_oo0, orbcs, orbvs.conj())
            v1mo_oooo0 = lib.einsum('xpq,qo,pv->xov', v1ao_oo0, orbos, orbos.conj())
            v1mo_ovoo0 = lib.einsum('xpq,qo,pv->xov', v1ao_oo0, orbos, orbvs.conj())

            v1mo_coov0 = lib.einsum('xpq,qo,pv->xov', v1ao_ov0, orbcs, orbos.conj())
            v1mo_cvov0 = lib.einsum('xpq,qo,pv->xov', v1ao_ov0, orbcs, orbvs.conj())
            v1mo_ooov0 = lib.einsum('xpq,qo,pv->xov', v1ao_ov0, orbos, orbos.conj())
            v1mo_ovov0 = lib.einsum('xpq,qo,pv->xov', v1ao_ov0, orbos, orbvs.conj())

            v1mo_coco1 = lib.einsum('xpq,qo,pv->xov', v1ao_co1, orbcs, orbos.conj())
            v1mo_coov1 = lib.einsum('xpq,qo,pv->xov', v1ao_ov1, orbcs, orbos.conj())
            v1mo_ovco1 = lib.einsum('xpq,qo,pv->xov', v1ao_co1, orbos, orbvs.conj())
            v1mo_ovov1 = lib.einsum('xpq,qo,pv->xov', v1ao_ov1, orbos, orbvs.conj())

            v1mo_co = np.zeros_like(zs_co)
            v1mo_cv = np.zeros_like(zs_cv)
            v1mo_oo = np.zeros_like(zs_oo)
            v1mo_ov = np.zeros_like(zs_ov)

            v1mo_co += v1mo_coco0 + v1mo_coco1 / (2 * si - 1) + v1mo_cocv0 * np.sqrt((2 * si + 1) / (2 * si))
            v1mo_co += (
                v1mo_cooo0 * np.sqrt(2 * si / (2 * si - 1))
                + v1mo_coov0 * (2 * si / (2 * si - 1))
                - v1mo_coov1 / (2 * si - 1)
            )
            v1mo_cv += (
                v1mo_cvco0 * np.sqrt((2 * si + 1) / (2 * si))
                + v1mo_cvcv0
                + v1mo_cvoo0 * np.sqrt((2 * si + 1) / (2 * si - 1))
            )
            v1mo_cv += v1mo_cvov0 * np.sqrt((2 * si + 1) / (2 * si))
            v1mo_oo += v1mo_ooco0 * np.sqrt(2 * si / (2 * si - 1)) + v1mo_oocv0 * np.sqrt((2 * si + 1) / (2 * si - 1))
            v1mo_oo += v1mo_oooo0 + v1mo_ooov0 * np.sqrt(2 * si / (2 * si - 1))
            v1mo_ov += (
                v1mo_ovco0 * (2 * si / (2 * si - 1))
                - v1mo_ovco1 / (2 * si - 1)
                + v1mo_ovcv0 * np.sqrt((2 * si + 1) / (2 * si))
            )
            v1mo_ov += v1mo_ovoo0 * np.sqrt(2 * si / (2 * si - 1)) + v1mo_ovov0 + v1mo_ovov1 / (2 * si - 1)

            v1mo_co += lib.einsum('ij,uv,xjv->xiu', np.eye(ncs), fockb_oo, zs_co)
            v1mo_co -= lib.einsum('uv,ji,xjv->xiu', np.eye(nos), focka_cc, zs_co)
            v1mo_co += lib.einsum('uv,ji,xjv->xiu', np.eye(nos), focks_cc, zs_co) * 2 / (2 * si - 1)
            v1mo_co += lib.einsum('ij,ub,xjb->xiu', np.eye(ncs), fockb_ov, zs_cv) * np.sqrt((2 * si + 1) / (2 * si))
            v1mo_co -= lib.einsum('vu,wi,xwv->xiu', np.eye(nos), focka_oc, zs_oo) * np.sqrt(2 * si / (2 * si - 1))
            v1mo_co += lib.einsum('vw,ui,xwv->xiu', np.eye(nos), fockb_oc, zs_oo) / np.sqrt(2 * si * (2 * si - 1))

            v1mo_cv += lib.einsum('ij,av,xjv->xia', np.eye(ncs), fockb_ov.conj().T, zs_co) * np.sqrt(
                (2 * si + 1) / (2 * si)
            )
            v1mo_cv += lib.einsum('ij,ab,xjb->xia', np.eye(ncs), fockb_vv, zs_cv)
            v1mo_cv -= lib.einsum('ab,ji,xjb->xia', np.eye(nvs), focka_cc, zs_cv)
            v1mo_cv += lib.einsum('ij,ab,xjb->xia', np.eye(ncs), focks_vv, zs_cv) / si
            v1mo_cv += lib.einsum('ab,ji,xjb->xia', np.eye(nvs), focks_cc, zs_cv) / si
            v1mo_cv += lib.einsum('vw,ai,xwv->xia', np.eye(nos), focks_vc, zs_oo) * np.sqrt((2 * si + 1) / (2 * si - 1))
            v1mo_cv -= lib.einsum('ab,vi,xvb->xia', np.eye(nvs), focka_oc, zs_ov) * np.sqrt((2 * si + 1) / (2 * si))

            v1mo_oo -= lib.einsum('vt,ju,xjv->xut', np.eye(nos), focka_oc.conj().T, zs_co) * np.sqrt(
                2 * si / (2 * si - 1)
            )
            v1mo_oo += lib.einsum('ut,jv,xjv->xut', np.eye(nos), fockb_oc.conj().T, zs_co) / np.sqrt(
                2 * si * (2 * si - 1)
            )
            v1mo_oo += (
                lib.einsum('ut,jb,xjb->xut', np.eye(nos), focks_vc.conj().T, zs_cv)
                * np.sqrt((2 * si + 1) / (2 * si - 1))
                / si
            )
            v1mo_oo += lib.einsum('wu,tv,xwv->xut', np.eye(nos), fockb_oo, zs_oo)
            v1mo_oo -= lib.einsum('tv,uw,xwv->xut', np.eye(nos), focka_oo, zs_oo)
            v1mo_oo += lib.einsum('uv,tb,xvb->xut', np.eye(nos), fockb_ov, zs_ov) * np.sqrt(2 * si / (2 * si - 1))
            v1mo_oo -= lib.einsum('tu,vb,xvb->xut', np.eye(nos), focka_ov, zs_ov) / np.sqrt(2 * si * (2 * si - 1))

            v1mo_ov -= lib.einsum('ab,ju,xjb->xua', np.eye(nvs), focka_oc.conj().T, zs_cv) * np.sqrt(
                (2 * si + 1) / (2 * si)
            )
            v1mo_ov += lib.einsum('wu,av,xwv->xua', np.eye(nos), fockb_ov.conj().T, zs_oo) * np.sqrt(
                2 * si / (2 * si - 1)
            )
            v1mo_ov -= lib.einsum('vw,au,xwv->xua', np.eye(nos), focka_ov.conj().T, zs_oo) / np.sqrt(
                2 * si * (2 * si - 1)
            )
            v1mo_ov += lib.einsum('uv,ab,xvb->xua', np.eye(nos), fockb_vv, zs_ov)
            v1mo_ov -= lib.einsum('ab,vu,xvb->xua', np.eye(nvs), focka_oo, zs_ov)
            v1mo_ov += lib.einsum('uv,ab,xvb->xua', np.eye(nos), focks_vv, zs_ov) * 2 / (2 * si - 1)

            v1mo = np.zeros_like(zs)
            v1mo[:, :ncs, :nos] = v1mo_co
            v1mo[:, :ncs, nos:] = v1mo_cv
            v1mo[:, ncs:, :nos] = v1mo_oo
            v1mo[:, ncs:, nos:] = v1mo_ov
            return v1mo.reshape(len(v1mo), -1)

        return vind, hdiag

    def init_guess(self, mf=None, nstates=None, wfnsym=None):
        if nstates is None:
            nstates = self.nstates
        return self.sftda.init_guess(nstates=nstates)

    def kernel(self, x0=None, nstates=None, extype=None):
        """
        Spin-Flip TDA diagonalization solver
        """
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

        vind, hdiag = self.gen_vind()
        precond = self.get_precond(hdiag)

        x0sym = None
        if x0 is None:
            x0 = self.init_guess()

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

        nmo = self._scf.mo_occ[0].size
        nocca, noccb = self._scf.nelec
        nvirb = nmo - noccb

        self.xy = [(xi.reshape(nocca, nvirb), 0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDA_SF', *cpu0)
        self._finalize()
        return self.e, self.xy

    def get_a_sasf(self, mf=None, collinear_samples=None):
        if mf is None:
            mf = self._scf
        if collinear_samples is None:
            collinear_samples = self.collinear_samples
        return get_a_sasf(mf, collinear_samples=collinear_samples)


if __name__ == '__main__':
    mol = gto.M(atom='O; O 1 1.2', charge=0, spin=2, basis='ccpvtz')
    mf = mol.ROKS(xc='B3LYP')
    mf.kernel()
    td = TDA_SASF(mf)
    td.collinear_samples = 20
    td.nstates = 5
    td.verbose = 9
    td.kernel()
