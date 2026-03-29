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
# J. Chem. Phys. 2025, 163, 094111
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

def get_a_sasf(mf, collinear_samples=20):
    """
    A_[i,a,j,b]for A_abab
    """
    assert isinstance(mf, dft.roks.ROKS)
    mol = mf.mol
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
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
    focko = orbo.T @ focka @ orbo
    fockv = orbv.T @ fockb @ orbv

    a = np.zeros((nocc, nvir, nocc, nvir))

    a += np.einsum('ij,ab->iajb', np.eye(nocc), fockv)
    a -= np.einsum('ab,ij->iajb', np.eye(nvir), focko)

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
    a_ooco = a_cooo.transpose(2, 3, 0, 1)
    a_oocv = a_cvoo.transpose(2, 3, 0, 1)
    a_ooov = a_ovoo.transpose(2, 3, 0, 1)

    if collinear_samples > 0:
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        dm0 = mf.to_uks().make_rdm1()
        make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]
        xctype = ni._xc_type(mf.xc)
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory * 0.8 - mem_now)
        nimc = dft.numint2c.NumInt2C()
        nimc.collinear = 'mcol'
        nimc.collinear_samples = collinear_samples
        eval_xc_eff_sf = mcfun_eval_xc_adapter_sf(nimc, mf.xc)

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                fxc_sf = eval_xc_eff_sf(mf.xc, rho_z, deriv=2, xctype=xctype)[2]
                fxc_sc = ni.eval_xc_eff(mf.xc, (rho0a, rho0b), deriv=2, xctype=xctype)[2]
                wfxc_sf = fxc_sf[0, 0] * weight
                wfxc_sc = fxc_sc[:,0,:,0] * weight
                rho_cs = lib.einsum('rp,pi->ri', ao, orbcs)
                rho_os = lib.einsum('rp,pi->ri', ao, orbos)
                rho_vs = lib.einsum('rp,pi->ri', ao, orbvs)
                rho_co = np.einsum('ri,ra->ria', rho_cs, rho_os)
                rho_cv = np.einsum('ri,ra->ria', rho_cs, rho_vs)
                rho_oo = np.einsum('ri,ra->ria', rho_os, rho_os)
                rho_ov = np.einsum('ri,ra->ria', rho_os, rho_vs)

                w_cv_sf = np.einsum('ria,r->ria', rho_cv, wfxc_sf)
                w_co_sf = np.einsum('ria,r->ria', rho_co, wfxc_sf)
                w_oo_sf = np.einsum('ria,r->ria', rho_oo, wfxc_sf)
                w_ov_sf = np.einsum('ria,r->ria', rho_ov, wfxc_sf)

                a_coco += lib.einsum('riu,rjv->iujv', rho_co, w_co_sf) * 2 * si / (2 * si - 1)
                a_coov += lib.einsum('riu,rvb->iuvb', rho_co, w_ov_sf) * 2 * si / (2 * si - 1)
                a_cvco += lib.einsum('ria,rjv->iajv', rho_cv, w_co_sf) * 2 * si / (2 * si - 1) * np.sqrt((2 * si + 1) / (2 * si))
                a_cvcv += lib.einsum('ria,rjb->iajb', rho_cv, w_cv_sf) * (1 - 1 / si + 4 / (2 * si - 1))
                a_cvov += lib.einsum('ria,rvb->iavb', rho_cv, w_ov_sf) * 2 * si / (4 * si - 2) * np.sqrt((4 * si + 2) / si)
                a_ooco += lib.einsum('rut,rjv->utjv', rho_oo, w_co_sf) * np.sqrt(2 * si / (2 * si - 1))
                a_oocv += lib.einsum('rut,rjb->utjb', rho_oo, w_cv_sf) * np.sqrt((2 * si + 1) / (2 * si - 1))
                a_oooo += lib.einsum('rut,rwv->utwv', rho_oo, w_oo_sf)
                a_ooov += lib.einsum('rut,rvb->utvb', rho_oo, w_ov_sf) * np.sqrt(2 * si / (2 * si - 1))
                a_ovov += lib.einsum('rua,rvb->uavb', rho_ov, w_ov_sf) * 2 * si / (2 * si - 1)

                w_co_bbbb = np.einsum('ria,r->ria', rho_co, wfxc_sc[1,1])
                w_co_aabb = np.einsum('ria,r->ria', rho_co, wfxc_sc[0,1])
                w_cv_t = np.einsum('ria,r->ria', rho_cv, wfxc_sc[0,0]-wfxc_sc[0,1]-wfxc_sc[1,0]+wfxc_sc[1,1]) * 0.5
                w_ov_aaaa = np.einsum('ria,r->ria', rho_ov, wfxc_sc[0,0])
                w_ov_bbaa = np.einsum('ria,r->ria', rho_ov, wfxc_sc[1,0])

                a_coco -= lib.einsum('riu,rjv->iujv', rho_co, w_co_bbbb) / (2 * si - 1)
                a_coov += lib.einsum('riu,rvb->iuvb', rho_co, w_ov_bbaa) / (2 * si - 1)
                a_cvco += lib.einsum('ria,rjv->iajv', rho_cv, w_co_aabb-w_co_bbbb) / (2 * si - 1) * np.sqrt((2 * si + 1) / (2 * si))
                a_cvcv += lib.einsum('ria,rjb->iajb', rho_cv, w_cv_t) * (1 / si - 4 / (2 * si - 1))
                a_cvov += lib.einsum('ria,rvb->iavb', rho_cv, -w_ov_aaaa+w_ov_bbaa) / (4 * si - 2) * np.sqrt((4 * si + 2) / si)
                a_ovov -= lib.einsum('rua,rvb->uavb', rho_ov, w_ov_aaaa) / (2 * si - 1)
        elif xctype == 'HF':
            pass
        else:
            raise NotImplementedError('Only LDA is implemented for Lucas Visscher SASF-TDA.')


    # HF
    if hyb > 0:
        eri_coco1 = ao2mo.general(mol, [orbos, orbcs, orbcs, orbos], compact=False).reshape(nos, ncs, ncs, nos)
        eri_coco2 = ao2mo.general(mol, [orbos, orbos, orbcs, orbcs], compact=False).reshape(nos, nos, ncs, ncs)
        a_coco -= np.einsum('uijv->iujv', eri_coco1) / (2 * si - 1)
        a_coco -= np.einsum('uvji->iujv', eri_coco2) * hyb
        eri_coov1 = ao2mo.general(mol, [orbos, orbcs, orbos, orbvs], compact=False).reshape(nos, ncs, nos, nvs)
        eri_coov2 = ao2mo.general(mol, [orbos, orbvs, orbos, orbcs], compact=False).reshape(nos, nvs, nos, ncs)
        a_coov += np.einsum('uivb->iuvb', eri_coov1) / (2 * si - 1)
        a_coov -= np.einsum('ubvi->iuvb', eri_coov2) * hyb * 2 * si / (2 * si - 1)
        eri_cvco = ao2mo.general(mol, [orbvs, orbos, orbcs, orbcs], compact=False).reshape(nvs, nos, ncs, ncs)
        a_cvco -= np.einsum('avji->iajv', eri_cvco) * hyb * np.sqrt((2 * si + 1) / (2 * si))
        eri_cvcv = ao2mo.general(mol, [orbvs, orbvs, orbcs, orbcs], compact=False).reshape(nvs, nvs, ncs, ncs)
        a_cvcv -= np.einsum('abji->iajb', eri_cvcv) * hyb
        eri_cvov = ao2mo.general(mol, [orbvs, orbvs, orbos, orbcs], compact=False).reshape(nvs, nvs, nos, ncs)
        a_cvov -= np.einsum('abvi->iavb', eri_cvov) * hyb * np.sqrt((2 * si + 1) / (2 * si))
        eri_ovov1 = ao2mo.general(mol, [orbvs, orbos, orbos, orbvs], compact=False).reshape(nvs, nos, nos, nvs)
        eri_ovov2 = ao2mo.general(mol, [orbvs, orbvs, orbos, orbos], compact=False).reshape(nvs, nvs, nos, nos)
        a_ovov -= np.einsum('auvb->uavb', eri_ovov1) / (2 * si - 1)
        a_ovov -= np.einsum('abvu->uavb', eri_ovov2) * hyb
        eri_ooco = ao2mo.general(mol, [orbos, orbos, orbcs, orbos], compact=False).reshape(nos, nos, ncs, nos)
        a_ooco -= np.einsum('tvju->utjv', eri_ooco) * hyb * np.sqrt(2 * si / (2 * si - 1))
        eri_oocv = ao2mo.general(mol, [orbos, orbvs, orbcs, orbos], compact=False).reshape(nos, nvs, ncs, nos)
        a_oocv -= np.einsum('tbju->utjb', eri_oocv) * hyb * np.sqrt((2 * si + 1) / (2 * si - 1))
        eri_oooo = ao2mo.general(mol, [orbos, orbos, orbos, orbos], compact=False).reshape(nos, nos, nos, nos)
        a_oooo -= np.einsum('tvwu->utwv', eri_oooo) * hyb
        eri_ooov = ao2mo.general(mol, [orbos, orbvs, orbos, orbos], compact=False).reshape(nos, nvs, nos, nos)
        a_ooov -= np.einsum('tbuv->utvb', eri_ooov) * hyb * np.sqrt(2 * si / (2 * si - 1))


    a_cooo = a_ooco.transpose(2, 3, 0, 1)
    a_cvoo = a_oocv.transpose(2, 3, 0, 1)
    a_ovoo = a_ooov.transpose(2, 3, 0, 1)

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