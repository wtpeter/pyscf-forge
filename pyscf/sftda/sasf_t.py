import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf import dft
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.tdscf.uhf import TDBase
from pyscf import __config__
from pyscf.sftda.numint2c_sftd import mcfun_eval_xc_adapter_sf
from pyscf.sftda.numint2c_sftd import cache_xc_kernel_sf
from pyscf.sftda.uhf_sf import TDA_SF
from pyscf.dft.gen_grid import NBINS
from pyscf.dft.numint import _scale_ao_sparse, _dot_ao_ao_sparse, _dot_ao_dm_sparse, _contract_rho_sparse
from pyscf.tdscf._lr_eig import eigh as lr_eigh

def get_a_sasf_m0(mf, mo_coeff=None, mo_occ=None, collinear_samples=20, with_oo=True, Dz0=False):
    # assert isinstance(mf, dft.roks.ROKS)
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    
    mol = mf.mol
    nao, nmo = mo_coeff.shape
    si = (mol.nelec[0] - mol.nelec[1]) * 0.5
    # assert si >= 1, 'SASFTDA only supports case that Sf=Si>=1.'

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
    fock0 = 0.5 * (focka + fockb)
    fockz = 0.5 * (focka - fockb)

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    a_cv0cv0 = np.zeros((ncs, nvs, ncs, nvs))
    a_cvcv = np.zeros((ncs, nvs, ncs, nvs))
    a_coco = np.zeros((ncs, nos, ncs, nos))
    a_ovov = np.zeros((nos, nvs, nos, nvs))
    a_cvco = np.zeros((ncs, nvs, ncs, nos))
    a_cvov = np.zeros((ncs, nvs, nos, nvs))
    a_coov = np.zeros((ncs, nos, nos, nvs))
    a_cv0cv = np.zeros((ncs, nvs, ncs, nvs))
    a_cv0co = np.zeros((ncs, nvs, ncs, nos))
    a_cv0ov = np.zeros((ncs, nvs, nos, nvs))

    fockz_cc = orbcs.T @ fockz @ orbcs
    fockz_vv = orbvs.T @ fockz @ orbvs
    fockb_vo = orbvs.T @ fockb @ orbos
    focka_oc = orbos.T @ focka @ orbcs
    fockb_oo = orbos.T @ fockb @ orbos
    fockb_cc = orbcs.T @ fockb @ orbcs
    focka_vv = orbvs.T @ focka @ orbvs
    focka_oo = orbos.T @ focka @ orbos
    fock0_vv = orbvs.T @ fock0 @ orbvs
    fock0_cc = orbcs.T @ fock0 @ orbcs

    a_cv0cv0 += lib.einsum('ij,ab->iajb', np.eye(ncs), fock0_vv)
    a_cv0cv0 -= lib.einsum('ab,ji->iajb', np.eye(nvs), fock0_cc)
    a_cvcv = a_cv0cv0.copy()
    a_cvcv -= lib.einsum('ij,ab->iajb', np.eye(ncs), fockz_vv) / si
    a_cvcv -= lib.einsum('ab,ji->iajb', np.eye(nvs), fockz_cc) / si
    a_cvco += lib.einsum('ij,av->iajv', np.eye(ncs), fockb_vo) * np.sqrt((si + 1) / (2 * si))
    a_cvov -= lib.einsum('ab,vi->iavb', np.eye(nvs), focka_oc) * np.sqrt((si + 1) / (2 * si))
    a_coco += lib.einsum('ij,uv->iujv', np.eye(ncs), fockb_oo)
    a_coco -= lib.einsum('uv,ji->iujv', np.eye(nos), fockb_cc)
    a_ovov += lib.einsum('uv,ab->uavb', np.eye(nos), focka_vv)
    a_ovov -= lib.einsum('ab,vu->uavb', np.eye(nvs), focka_oo) 
    a_cv0cv -= lib.einsum('ij,ab->iajb', np.eye(ncs), fockz_vv) * np.sqrt((si + 1) / si)
    a_cv0cv += lib.einsum('ab,ji->iajb', np.eye(nvs), fockz_cc) * np.sqrt((si + 1) / si)
    a_cv0co += lib.einsum('ij,av->iajv', np.eye(ncs), fockb_vo) * np.sqrt(0.5)
    a_cv0ov += lib.einsum('ab,vi->iavb', np.eye(nvs), focka_oc) * np.sqrt(0.5)

    # J part
    eri = ao2mo.general(mol, (orbvs, orbcs, orbcs, orbvs), compact=False).reshape(nvs, ncs, ncs, nvs)
    a_cv0cv0 += np.einsum('aijb->iajb', eri) * 2
    eri = ao2mo.general(mol, (orbos, orbcs, orbcs, orbos), compact=False).reshape(nos, ncs, ncs, nos)
    a_coco += np.einsum('uijv->iujv', eri)
    eri = ao2mo.general(mol, (orbvs, orbos, orbos, orbvs), compact=False).reshape(nvs, nos, nos, nvs)
    a_ovov += np.einsum('auvb->uavb', eri)
    eri = ao2mo.general(mol, (orbos, orbcs, orbos, orbvs), compact=False).reshape(nos, ncs, nos, nvs)
    a_coov -= np.einsum('uivb->iuvb', eri)
    eri = ao2mo.general(mol, (orbvs, orbcs, orbcs, orbos), compact=False).reshape(nvs, ncs, ncs, nos)
    a_cv0co += np.einsum('aijv->iajv', eri) * np.sqrt(2)
    eri = ao2mo.general(mol, (orbvs, orbcs, orbos, orbvs), compact=False).reshape(nvs, ncs, nos, nvs)
    a_cv0ov -= np.einsum('aivb->iavb', eri) * np.sqrt(2)

    # K part
    if hybrid:
        eri = ao2mo.general(mol, (orbvs, orbvs, orbcs, orbcs), compact=False).reshape(nvs, nvs, ncs, ncs)
        a_cv0cv0 -= np.einsum('abji->iajb', eri) * hyb
        a_cvcv -= np.einsum('abji->iajb', eri) * hyb
        eri = ao2mo.general(mol, (orbvs, orbos, orbcs, orbcs), compact=False).reshape(nvs, nos, ncs, ncs)
        a_cvco -= np.einsum('avji->iajv', eri) * hyb * np.sqrt((si + 1) / (2 * si))
        a_cv0co -= np.einsum('avji->iajv', eri) * hyb * np.sqrt(0.5)
        eri = ao2mo.general(mol, (orbvs, orbvs, orbos, orbcs), compact=False).reshape(nvs, nvs, nos, ncs)
        a_cvov -= np.einsum('abvi->iavb', eri) * hyb * np.sqrt((si + 1) / (2 * si))
        a_cv0ov += np.einsum('abvi->iavb', eri) * hyb * np.sqrt(0.5)
        eri = ao2mo.general(mol, (orbos, orbos, orbcs, orbcs), compact=False).reshape(nos, nos, ncs, ncs)
        a_coco -= np.einsum('uvji->iujv', eri) * hyb
        eri = ao2mo.general(mol, (orbvs, orbvs, orbos, orbos), compact=False).reshape(nvs, nvs, nos, nos)
        a_ovov -= np.einsum('abvu->uavb', eri) * hyb
    if omega != 0:
        with mol.with_range_coulomb(omega):
            k_fac = alpha - hyb
            eri = ao2mo.general(mol, (orbvs, orbvs, orbcs, orbcs), compact=False).reshape(nvs, nvs, ncs, ncs)
            a_cv0cv0 -= np.einsum('abji->iajb', eri) * k_fac
            a_cvcv -= np.einsum('abji->iajb', eri) * k_fac
            eri = ao2mo.general(mol, (orbvs, orbos, orbcs, orbcs), compact=False).reshape(nvs, nos, ncs, ncs)
            a_cvco -= np.einsum('avji->iajv', eri) * k_fac * np.sqrt((si + 1) / (2 * si))
            a_cv0co -= np.einsum('avji->iajv', eri) * k_fac * np.sqrt(0.5)
            eri = ao2mo.general(mol, (orbvs, orbvs, orbos, orbcs), compact=False).reshape(nvs, nvs, nos, ncs)
            a_cvov -= np.einsum('abvi->iavb', eri) * k_fac * np.sqrt((si + 1) / (2 * si))
            a_cv0ov += np.einsum('abvi->iavb', eri) * k_fac * np.sqrt(0.5)
            eri = ao2mo.general(mol, (orbos, orbos, orbcs, orbcs), compact=False).reshape(nos, nos, ncs, ncs)
            a_coco -= np.einsum('uvji->iujv', eri) * k_fac
            eri = ao2mo.general(mol, (orbvs, orbvs, orbos, orbos), compact=False).reshape(nvs, nvs, nos, nos)
            a_ovov -= np.einsum('abvu->uavb', eri) * k_fac

    # XC part
    dm0 = mf.to_uks().make_rdm1()
    make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]
    xctype = ni._xc_type(mf.xc)
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory * 0.8 - mem_now)
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
            rho0a = make_rho(0, ao, mask, xctype)
            rho0b = make_rho(1, ao, mask, xctype)
            if Dz0:
                rho0a = rho0b = 0.5 * (rho0a + rho0b)
            fxc_sc = ni.eval_xc_eff(mf.xc, (rho0a, rho0b), deriv=2, xctype=xctype)[2]
            wfxc_sc = fxc_sc[:,0,:,0] * weight

            rho_c = lib.einsum('rp,pi->ri', ao, orbcs)
            rho_o = lib.einsum('rp,pi->ri', ao, orbos)
            rho_v = lib.einsum('rp,pi->ri', ao, orbvs)
            rho_ov = np.einsum('ri,ra->ria', rho_o, rho_v)
            rho_co = np.einsum('ri,ra->ria', rho_c, rho_o)
            rho_cv = np.einsum('ri,ra->ria', rho_c, rho_v)

            w_cv = np.einsum('ria,r->ria', rho_cv, (wfxc_sc[0, 0] + wfxc_sc[0, 1] + wfxc_sc[1, 0] + wfxc_sc[1, 1]) * 0.5)
            iajb = lib.einsum('ria,rjb->iajb', rho_cv, w_cv)
            a_cv0cv0 += iajb

            w_cv = np.einsum('ria,r->ria', rho_cv, (wfxc_sc[0, 0] - wfxc_sc[0, 1] - wfxc_sc[1, 0] + wfxc_sc[1, 1]) * 0.5)
            iajb = lib.einsum('ria,rjb->iajb', rho_cv, w_cv)
            a_cvcv += iajb * (1 + 1 / si)

            w_co = np.einsum('ria,r->ria', rho_co, - wfxc_sc[0, 1] + wfxc_sc[1, 1])
            iajb = lib.einsum('ria,rjb->iajb', rho_cv, w_co)
            a_cvco += iajb * np.sqrt((si + 1) / (2 * si))

            w_ov = np.einsum('ria,r->ria', rho_ov, wfxc_sc[0, 0] - wfxc_sc[1, 0])
            iajb = lib.einsum('ria,rjb->iajb', rho_cv, w_ov)
            a_cvov += iajb * np.sqrt((si + 1) / (2 * si))

            w_co = np.einsum('ria,r->ria', rho_co, wfxc_sc[1, 1])
            iajb = lib.einsum('ria,rjb->iajb', rho_co, w_co)
            a_coco += iajb

            w_ov = np.einsum('ria,r->ria', rho_ov, wfxc_sc[1, 0])
            iajb = lib.einsum('ria,rjb->iajb', rho_co, w_ov)
            a_coov -= iajb

            w_ov = np.einsum('ria,r->ria', rho_ov, wfxc_sc[0, 0])
            iajb = lib.einsum('ria,rjb->iajb', rho_ov, w_ov)
            a_ovov += iajb

            w_cv = np.einsum('ria,r->ria', rho_cv,(wfxc_sc[0, 0] + wfxc_sc[0, 1] - wfxc_sc[1, 0] - wfxc_sc[1, 1]) * 0.5)
            iajb = lib.einsum('ria,rjb->iajb', rho_cv, w_cv)
            a_cv0cv -= iajb * np.sqrt((si + 1) / si)

            w_co = np.einsum('ria,r->ria', rho_co, wfxc_sc[0, 1] + wfxc_sc[1, 1])
            iajb = lib.einsum('ria,rjb->iajb', rho_cv, w_co)
            a_cv0co += iajb * np.sqrt(0.5)

            w_ov = np.einsum('ria,r->ria', rho_ov, wfxc_sc[0, 0] + wfxc_sc[1, 0])
            iajb = lib.einsum('ria,rjb->iajb', rho_cv, w_ov)
            a_cv0ov -= iajb * np.sqrt(0.5)

            if collinear_samples > 0:
                nimc = dft.numint2c.NumInt2C()
                nimc.collinear = 'mcol'
                nimc.collinear_samples = collinear_samples
                eval_xc_eff_sf = mcfun_eval_xc_adapter_sf(nimc, mf.xc)
                rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                fxc_sf = 2 * eval_xc_eff_sf(mf.xc, rho_z, deriv=2, xctype=xctype)[2]
                wfxc_sf = fxc_sf[0, 0] * weight
                w_cv = np.einsum('ria,r->ria', rho_cv, wfxc_sf)
                iajb = lib.einsum('ria,rjb->iajb', rho_cv, w_cv)
                a_cvcv -= iajb * (1 / si)

    elif xctype == 'GGA':
        ao_deriv = 1
        def make_pair_gga(r1, r2):
            r12 = np.einsum('xri,rj->xrij', r1, r2[0])
            r12[1:4] += np.einsum('ri,xrj->xrij', r1[0], r2[1:4])
            return r12
        for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
            rho0a = make_rho(0, ao, mask, xctype)
            rho0b = make_rho(1, ao, mask, xctype)
            if Dz0:
                rho0a = rho0b = 0.5 * (rho0a + rho0b)
            # fxc_sc shape: (2, 4, 2, 4, ngrids)
            fxc_sc = ni.eval_xc_eff(mf.xc, (rho0a, rho0b), deriv=2, xctype=xctype)[2]
            wfxc_sc = fxc_sc * weight

            rho_c = lib.einsum('xrp,pi->xri', ao, orbcs)
            rho_o = lib.einsum('xrp,pi->xri', ao, orbos)
            rho_v = lib.einsum('xrp,pi->xri', ao, orbvs)
            rho_ov = make_pair_gga(rho_o, rho_v)
            rho_co = make_pair_gga(rho_c, rho_o)
            rho_cv = make_pair_gga(rho_c, rho_v)

            v_cv0cv0 = (wfxc_sc[0,:,0] + wfxc_sc[0,:,1] + wfxc_sc[1,:,0] + wfxc_sc[1,:,1]) * 0.5
            w_cv = np.einsum('xyr,xria->yria', v_cv0cv0, rho_cv)
            a_cv0cv0 += lib.einsum('xria,xrjb->iajb', w_cv, rho_cv)

            v_cvcv = (wfxc_sc[0,:,0] - wfxc_sc[0,:,1] - wfxc_sc[1,:,0] + wfxc_sc[1,:,1]) * 0.5
            w_cv = np.einsum('xyr,xria->yria', v_cvcv, rho_cv)
            a_cvcv += lib.einsum('xria,xrjb->iajb', w_cv, rho_cv) * (1 + 1 / si)

            v_cvco = -wfxc_sc[0,:,1] + wfxc_sc[1,:,1]
            w_co = np.einsum('xyr,yrjv->xrjv', v_cvco, rho_co)
            a_cvco += lib.einsum('xria,xrjv->iajv', rho_cv, w_co) * np.sqrt((si + 1) / (2 * si))

            v_cvov = wfxc_sc[0,:,0] - wfxc_sc[1,:,0]
            w_ov = np.einsum('xyr,yrvb->xrvb', v_cvov, rho_ov)
            a_cvov += lib.einsum('xria,xrvb->iavb', rho_cv, w_ov) * np.sqrt((si + 1) / (2 * si))

            v_coco = wfxc_sc[1,:,1]
            w_co = np.einsum('xyr,xria->yria', v_coco, rho_co)
            a_coco += lib.einsum('xria,xrjb->iajb', w_co, rho_co)

            v_coov = wfxc_sc[1,:,0]
            w_ov = np.einsum('xyr,yrvb->xrvb', v_coov, rho_ov)
            a_coov -= lib.einsum('xriu,xrvb->iuvb', rho_co, w_ov)

            v_ovov = wfxc_sc[0,:,0]
            w_ov = np.einsum('xyr,xria->yria', v_ovov, rho_ov)
            a_ovov += lib.einsum('xria,xrjb->iajb', w_ov, rho_ov)

            v_cv0cv = (wfxc_sc[0,:,0] + wfxc_sc[0,:,1] - wfxc_sc[1,:,0] - wfxc_sc[1,:,1]) * 0.5
            w_cv = np.einsum('xyr,xria->yria', v_cv0cv, rho_cv)
            a_cv0cv -= lib.einsum('xria,xrjb->iajb', w_cv, rho_cv) * np.sqrt((si + 1) / si)

            v_cv0co = wfxc_sc[0,:,1] + wfxc_sc[1,:,1]
            w_co = np.einsum('xyr,yrjv->xrjv', v_cv0co, rho_co)
            a_cv0co += lib.einsum('xria,xrjv->iajv', rho_cv, w_co) * np.sqrt(0.5)

            v_cv0ov = wfxc_sc[0,:,0] + wfxc_sc[1,:,0]
            w_ov = np.einsum('xyr,yrvb->xrvb', v_cv0ov, rho_ov)
            a_cv0ov -= lib.einsum('xria,xrvb->iavb', rho_cv, w_ov) * np.sqrt(0.5)

            if collinear_samples > 0:
                nimc = dft.numint2c.NumInt2C()
                nimc.collinear = 'mcol'
                nimc.collinear_samples = collinear_samples
                eval_xc_eff_sf = mcfun_eval_xc_adapter_sf(nimc, mf.xc)
                rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                # SF 不需要提取 [0,0]，直接返回 (4, 4, ngrids)
                fxc_sf = 2 * eval_xc_eff_sf(mf.xc, rho_z, deriv=2, xctype=xctype)[2]
                wfxc_sf = fxc_sf * weight
                w_cv = np.einsum('xyr,xria->yria', wfxc_sf, rho_cv)
                iajb = lib.einsum('xria,xrjb->iajb', w_cv, rho_cv)
                a_cvcv -= iajb * (1 / si)

    elif xctype == 'MGGA':
        ao_deriv = 1
        def make_pair_mgga(r1, r2):
            r12 = np.einsum('xri,rj->xrij', r1, r2[0])
            r12[1:4] += np.einsum('ri,xrj->xrij', r1[0], r2[1:4])
            tau12 = np.einsum('xri,xrj->rij', r1[1:4], r2[1:4]) * 0.5
            return np.vstack([r12, tau12[np.newaxis]])
            
        for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
            rho0a = make_rho(0, ao, mask, xctype)
            rho0b = make_rho(1, ao, mask, xctype)
            if Dz0:
                rho0a = rho0b = 0.5 * (rho0a + rho0b)

            # fxc_sc shape: (2, 5, 2, 5, ngrids)
            fxc_sc = ni.eval_xc_eff(mf.xc, (rho0a, rho0b), deriv=2, xctype=xctype)[2]
            wfxc_sc = fxc_sc * weight

            rho_c = lib.einsum('xrp,pi->xri', ao, orbcs)
            rho_o = lib.einsum('xrp,pi->xri', ao, orbos)
            rho_v = lib.einsum('xrp,pi->xri', ao, orbvs)
            rho_ov = make_pair_mgga(rho_o, rho_v)
            rho_co = make_pair_mgga(rho_c, rho_o)
            rho_cv = make_pair_mgga(rho_c, rho_v)

            v_cv0cv0 = (wfxc_sc[0,:,0] + wfxc_sc[0,:,1] + wfxc_sc[1,:,0] + wfxc_sc[1,:,1]) * 0.5
            w_cv = np.einsum('xyr,xria->yria', v_cv0cv0, rho_cv)
            a_cv0cv0 += lib.einsum('xria,xrjb->iajb', w_cv, rho_cv)

            v_cvcv = (wfxc_sc[0,:,0] - wfxc_sc[0,:,1] - wfxc_sc[1,:,0] + wfxc_sc[1,:,1]) * 0.5
            w_cv = np.einsum('xyr,xria->yria', v_cvcv, rho_cv)
            a_cvcv += lib.einsum('xria,xrjb->iajb', w_cv, rho_cv) * (1 + 1 / si)

            v_cvco = -wfxc_sc[0,:,1] + wfxc_sc[1,:,1]
            w_co = np.einsum('xyr,yrjv->xrjv', v_cvco, rho_co)
            a_cvco += lib.einsum('xria,xrjv->iajv', rho_cv, w_co) * np.sqrt((si + 1) / (2 * si))

            v_cvov = wfxc_sc[0,:,0] - wfxc_sc[1,:,0]
            w_ov = np.einsum('xyr,yrvb->xrvb', v_cvov, rho_ov)
            a_cvov += lib.einsum('xria,xrvb->iavb', rho_cv, w_ov) * np.sqrt((si + 1) / (2 * si))

            v_coco = wfxc_sc[1,:,1]
            w_co = np.einsum('xyr,xria->yria', v_coco, rho_co)
            a_coco += lib.einsum('xria,xrjb->iajb', w_co, rho_co)

            v_coov = wfxc_sc[1,:,0]
            w_ov = np.einsum('xyr,yrvb->xrvb', v_coov, rho_ov)
            a_coov -= lib.einsum('xriu,xrvb->iuvb', rho_co, w_ov)

            v_ovov = wfxc_sc[0,:,0]
            w_ov = np.einsum('xyr,xria->yria', v_ovov, rho_ov)
            a_ovov += lib.einsum('xria,xrjb->iajb', w_ov, rho_ov)

            v_cv0cv = (wfxc_sc[0,:,0] + wfxc_sc[0,:,1] - wfxc_sc[1,:,0] - wfxc_sc[1,:,1]) * 0.5
            w_cv = np.einsum('xyr,xria->yria', v_cv0cv, rho_cv)
            a_cv0cv -= lib.einsum('xria,xrjb->iajb', w_cv, rho_cv) * np.sqrt((si + 1) / si)

            v_cv0co = wfxc_sc[0,:,1] + wfxc_sc[1,:,1]
            w_co = np.einsum('xyr,yrjv->xrjv', v_cv0co, rho_co)
            a_cv0co += lib.einsum('xria,xrjv->iajv', rho_cv, w_co) * np.sqrt(0.5)

            v_cv0ov = wfxc_sc[0,:,0] + wfxc_sc[1,:,0]
            w_ov = np.einsum('xyr,yrvb->xrvb', v_cv0ov, rho_ov)
            a_cv0ov -= lib.einsum('xria,xrvb->iavb', rho_cv, w_ov) * np.sqrt(0.5)

            if collinear_samples > 0:
                nimc = dft.numint2c.NumInt2C()
                nimc.collinear = 'mcol'
                nimc.collinear_samples = collinear_samples
                eval_xc_eff_sf = mcfun_eval_xc_adapter_sf(nimc, mf.xc)
                rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                fxc_sf = 2 * eval_xc_eff_sf(mf.xc, rho_z, deriv=2, xctype=xctype)[2]
                wfxc_sf = fxc_sf * weight
                w_cv = np.einsum('xyr,xria->yria', wfxc_sf, rho_cv)
                iajb = lib.einsum('xria,xrjb->iajb', w_cv, rho_cv)
                a_cvcv -= iajb * (1 / si)

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'{xctype} not supported for SASF-TDA.')

    # OO part
    fockz_cv = orbcs.T @ fockz @ orbvs
    fockb_co = orbcs.T @ fockb @ orbos
    focka_ov = orbos.T @ focka @ orbvs
    fock0_cv = orbcs.T @ fock0 @ orbvs
    
    a_cvoo = fockz_cv * 2 * np.sqrt((si + 1) / (2 * si))
    a_cooo = - fockb_co
    a_ooov = focka_ov
    a_cv0oo = - fock0_cv * np.sqrt(2)
    if not with_oo:
        a_cvoo = np.zeros_like(a_cvoo)
        a_cooo = np.zeros_like(a_cooo)
        a_ooov = np.zeros_like(a_ooov)
        a_cv0oo = np.zeros_like(a_cv0oo)

    # Fill the Whole matrix
    # CO | CV | OO | OV | CV0
    co = ncs * nos
    cv = ncs * nvs
    ov = nos * nvs
    a = np.zeros((2 * cv + co + ov + 1, 2 * cv + co + ov + 1))

    a[:co, :co] = a_coco.reshape(co, co)
    a[:co, co:co+cv] = a_cvco.transpose(2, 3, 0, 1).reshape(co, cv)
    a[:co, co+cv:co+cv+1] = a_cooo.reshape(co, 1)
    a[:co, co+cv+1:co+cv+ov+1] = a_coov.reshape(co, ov)
    a[:co, co+cv+1+ov:] = a_cv0co.transpose(2, 3, 0, 1).reshape(co, cv)

    a[co:co+cv, :co] = a_cvco.reshape(cv, co)
    a[co:co+cv, co:co+cv] = a_cvcv.reshape(cv, cv)
    a[co:co+cv, co+cv:co+cv+1] = a_cvoo.reshape(cv, 1)
    a[co:co+cv, co+cv+1:co+cv+ov+1] = a_cvov.reshape(cv, ov)
    a[co:co+cv, co+cv+1+ov:] = a_cv0cv.transpose(2, 3, 0, 1).reshape(cv, cv)

    a[co+cv:co+cv+1, :co] = a_cooo.reshape(1, co)
    a[co+cv:co+cv+1, co:co+cv] = a_cvoo.reshape(1, cv)
    a[co+cv:co+cv+1, co+cv:co+cv+1] = 0.0
    a[co+cv:co+cv+1, co+cv+1:co+cv+ov+1] = a_ooov.reshape(1, ov)
    a[co+cv:co+cv+1, co+cv+1+ov:] = a_cv0oo.reshape(1, cv)

    a[co+cv+1:co+cv+ov+1, :co] = a_coov.transpose(2, 3, 0, 1).reshape(ov, co)
    a[co+cv+1:co+cv+ov+1, co:co+cv] = a_cvov.transpose(2, 3, 0, 1).reshape(ov, cv)
    a[co+cv+1:co+cv+ov+1, co+cv:co+cv+1] = a_ooov.reshape(ov, 1)
    a[co+cv+1:co+cv+ov+1, co+cv+1:co+cv+ov+1] = a_ovov.reshape(ov, ov)
    a[co+cv+1:co+cv+ov+1, co+cv+1+ov:] = a_cv0ov.transpose(2, 3, 0, 1).reshape(ov, cv)

    a[co+cv+1+ov:, :co] = a_cv0co.reshape(cv, co)
    a[co+cv+1+ov:, co:co+cv] = a_cv0cv.reshape(cv, cv)
    a[co+cv+1+ov:, co+cv:co+cv+1] = a_cv0oo.reshape(cv, 1)
    a[co+cv+1+ov:, co+cv+1:co+cv+ov+1] = a_cv0ov.reshape(cv, ov)
    a[co+cv+1+ov:, co+cv+1+ov:] = a_cv0cv0.reshape(cv, cv)

    assert abs(a - a.T).max() < 1e-10
    return a


def get_a_sasf_m1(mf, mo_coeff=None, mo_occ=None, collinear_samples=20, with_oo=True, Dz0=False):
    # assert isinstance(mf, dft.roks.ROKS)
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mol = mf.mol
    nao, nmo = mo_coeff.shape
    si = (mol.nelec[0] - mol.nelec[1]) * 0.5
    # assert si >= 1, 'SASFTDA only supports case that Sf=Si>=1.'

    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]
    orbcs = mo_coeff[:, csidx]
    orbos = mo_coeff[:, osidx]
    orbvs = mo_coeff[:, vsidx]
    ncs = orbcs.shape[1]
    nos = orbos.shape[1]
    nvs = orbvs.shape[1]

    fock = mf.get_fock()
    focka = fock.focka
    fockb = fock.fockb
    fock0 = 0.5 * (focka + fockb)
    fockz = 0.5 * (focka - fockb)

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    a_cv0cv0 = np.zeros((ncs, nvs, ncs, nvs))
    a_cvcv = np.zeros((ncs, nvs, ncs, nvs))
    a_coco = np.zeros((ncs, nos, ncs, nos))
    a_ovov = np.zeros((nos, nvs, nos, nvs))
    a_cvco = np.zeros((ncs, nvs, ncs, nos))
    a_cvov = np.zeros((ncs, nvs, nos, nvs))
    a_coov = np.zeros((ncs, nos, nos, nvs))
    a_cv0cv = np.zeros((ncs, nvs, ncs, nvs))
    a_cv0co = np.zeros((ncs, nvs, ncs, nos))
    a_cv0ov = np.zeros((ncs, nvs, nos, nvs))

    fockz_cc = orbcs.T @ fockz @ orbcs
    fockz_vv = orbvs.T @ fockz @ orbvs
    fockb_vo = orbvs.T @ fockb @ orbos
    focka_oc = orbos.T @ focka @ orbcs
    fockb_oo = orbos.T @ fockb @ orbos
    fockb_cc = orbcs.T @ fockb @ orbcs
    focka_vv = orbvs.T @ focka @ orbvs
    focka_oo = orbos.T @ focka @ orbos
    fock0_vv = orbvs.T @ fock0 @ orbvs
    fock0_cc = orbcs.T @ fock0 @ orbcs

    a_cv0cv0 += lib.einsum('ij,ab->iajb', np.eye(ncs), fock0_vv)
    a_cv0cv0 -= lib.einsum('ab,ji->iajb', np.eye(nvs), fock0_cc)
    a_cvcv = a_cv0cv0.copy()
    a_cvcv -= lib.einsum('ij,ab->iajb', np.eye(ncs), fockz_vv) / si
    a_cvcv -= lib.einsum('ab,ji->iajb', np.eye(nvs), fockz_cc) / si
    a_cvco += lib.einsum('ij,av->iajv', np.eye(ncs), fockb_vo) * np.sqrt((si + 1) / (2 * si))
    a_cvov -= lib.einsum('ab,vi->iavb', np.eye(nvs), focka_oc) * np.sqrt((si + 1) / (2 * si))
    a_coco += lib.einsum('ij,uv->iujv', np.eye(ncs), fockb_oo)
    a_coco -= lib.einsum('uv,ji->iujv', np.eye(nos), fockb_cc)
    a_ovov += lib.einsum('uv,ab->uavb', np.eye(nos), focka_vv)
    a_ovov -= lib.einsum('ab,vu->uavb', np.eye(nvs), focka_oo)
    a_cv0cv -= lib.einsum('ij,ab->iajb', np.eye(ncs), fockz_vv) * np.sqrt((si + 1) / si)
    a_cv0cv += lib.einsum('ab,ji->iajb', np.eye(nvs), fockz_cc) * np.sqrt((si + 1) / si)
    a_cv0co += lib.einsum('ij,av->iajv', np.eye(ncs), fockb_vo) * np.sqrt(0.5)
    a_cv0ov += lib.einsum('ab,vi->iavb', np.eye(nvs), focka_oc) * np.sqrt(0.5)

    def add_sf_hf(scale):
        # K^SF_{pq,rs} = -scale * (p r | s q) for exact exchange.
        eri = ao2mo.general(mol, (orbvs, orbvs, orbcs, orbcs), compact=False).reshape(nvs, nvs, ncs, ncs)
        a_cvcv[:] -= np.einsum('abji->iajb', eri) * scale
        a_cv0cv0[:] -= np.einsum('abji->iajb', eri) * scale

        eri = ao2mo.general(mol, (orbvs, orbcs, orbcs, orbvs), compact=False).reshape(nvs, ncs, ncs, nvs)
        a_cv0cv0[:] += np.einsum('aijb->iajb', eri) * (2 * scale)

        eri = ao2mo.general(mol, (orbvs, orbos, orbcs, orbcs), compact=False).reshape(nvs, nos, ncs, ncs)
        a_cvco[:] -= np.einsum('avji->iajv', eri) * scale * np.sqrt((si + 1) / (2 * si))
        a_cv0co[:] -= np.einsum('avji->iajv', eri) * scale * np.sqrt(0.5)

        eri = ao2mo.general(mol, (orbvs, orbcs, orbcs, orbos), compact=False).reshape(nvs, ncs, ncs, nos)
        a_cv0co[:] += np.einsum('aijv->iajv', eri) * (np.sqrt(2) * scale)

        eri = ao2mo.general(mol, (orbvs, orbvs, orbos, orbcs), compact=False).reshape(nvs, nvs, nos, ncs)
        a_cvov[:] -= np.einsum('abvi->iavb', eri) * scale * np.sqrt((si + 1) / (2 * si))
        a_cv0ov[:] += np.einsum('abvi->iavb', eri) * scale * np.sqrt(0.5)

        eri = ao2mo.general(mol, (orbvs, orbcs, orbos, orbvs), compact=False).reshape(nvs, ncs, nos, nvs)
        a_cv0ov[:] -= np.einsum('aivb->iavb', eri) * (np.sqrt(2) * scale)

        eri = ao2mo.general(mol, (orbos, orbos, orbcs, orbcs), compact=False).reshape(nos, nos, ncs, ncs)
        a_coco[:] -= np.einsum('uvji->iujv', eri) * scale

        eri = ao2mo.general(mol, (orbos, orbcs, orbcs, orbos), compact=False).reshape(nos, ncs, ncs, nos)
        a_coco[:] += np.einsum('uijv->iujv', eri) * scale

        eri = ao2mo.general(mol, (orbos, orbcs, orbos, orbvs), compact=False).reshape(nos, ncs, nos, nvs)
        a_coov[:] -= np.einsum('uivb->iuvb', eri) * scale

        eri = ao2mo.general(mol, (orbvs, orbvs, orbos, orbos), compact=False).reshape(nvs, nvs, nos, nos)
        a_ovov[:] -= np.einsum('abvu->uavb', eri) * scale

        eri = ao2mo.general(mol, (orbvs, orbos, orbos, orbvs), compact=False).reshape(nvs, nos, nos, nvs)
        a_ovov[:] += np.einsum('auvb->uavb', eri) * scale

    if hybrid:
        add_sf_hf(hyb)
    if omega != 0:
        with mol.with_range_coulomb(omega):
            add_sf_hf(alpha - hyb)

    dm0 = mf.to_uks().make_rdm1()
    make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]
    xctype = ni._xc_type(mf.xc)
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory * 0.8 - mem_now)

    def add_sf_xc_terms(rho_cv, rho_co, rho_ov, rho_cc, rho_oo, rho_vv, wfxc_sf):
        def contract(left, right):
            left_shape = left.shape[-2:]
            right_shape = right.shape[-2:]
            if left.ndim == 3:
                ngrids = left.shape[0]
                left2 = left.reshape(ngrids, -1)
                right2 = right.reshape(ngrids, -1)
                out = lib.einsum('gl,g,gm->lm', left2, wfxc_sf, right2)
            else:
                nvar, ngrids = left.shape[:2]
                left2 = left.reshape(nvar, ngrids, -1)
                right2 = right.reshape(nvar, ngrids, -1)
                out = lib.einsum('xgl,xyg,ygm->lm', left2, wfxc_sf, right2)
            return out.reshape(left_shape + right_shape)

        k_cv_cv = contract(rho_cv, rho_cv)  # i a, j b
        k_cv_co = contract(rho_cv, rho_co)  # i a, j v
        k_cv_ov = contract(rho_cv, rho_ov)  # i a, v b
        k_co_co = contract(rho_co, rho_co)  # i u, j v
        k_oo_cc = contract(rho_oo, rho_cc)  # u v, i j
        k_ov_co = contract(rho_ov, rho_co)  # u b, i v
        k_ov_ov = contract(rho_ov, rho_ov)  # u a, v b
        k_vv_oo = contract(rho_vv, rho_oo)  # a b, u v
        k_vv_cc = contract(rho_vv, rho_cc)  # a b, i j
        k_ov_cc = contract(rho_ov, rho_cc)  # v a, i j
        k_vv_co = contract(rho_vv, rho_co)  # a b, i v

        a_cvcv[:] += k_cv_cv
        a_cvco[:] += k_cv_co * np.sqrt((si + 1) / (2 * si))
        a_cvov[:] += k_cv_ov * np.sqrt((si + 1) / (2 * si))
        a_coco[:] += k_co_co
        a_coco[:] -= k_oo_cc.transpose(2, 0, 3, 1)
        a_coov[:] += k_ov_co.transpose(2, 0, 3, 1)
        a_ovov[:] += k_ov_ov
        a_ovov[:] -= k_vv_oo.transpose(2, 0, 3, 1)
        a_cv0cv0[:] += k_cv_cv
        a_cv0cv0[:] -= 2 * k_vv_cc.transpose(2, 0, 3, 1)
        a_cv0co[:] += k_cv_co * np.sqrt(0.5)
        a_cv0co[:] -= k_ov_cc.transpose(2, 1, 3, 0) * np.sqrt(2)
        a_cv0ov[:] -= k_cv_ov * np.sqrt(0.5)
        a_cv0ov[:] += k_vv_co.transpose(2, 0, 3, 1) * np.sqrt(2)

    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
            rho0a = make_rho(0, ao, mask, xctype)
            rho0b = make_rho(1, ao, mask, xctype)
            if Dz0:
                rho0a = rho0b = 0.5 * (rho0a + rho0b)
            rho_c = lib.einsum('rp,pi->ri', ao, orbcs)
            rho_o = lib.einsum('rp,pi->ri', ao, orbos)
            rho_v = lib.einsum('rp,pi->ri', ao, orbvs)
            rho_cv = np.einsum('ri,ra->ria', rho_c, rho_v)
            rho_co = np.einsum('ri,ru->riu', rho_c, rho_o)
            rho_ov = np.einsum('ru,ra->rua', rho_o, rho_v)
            rho_cc = np.einsum('ri,rj->rij', rho_c, rho_c)
            rho_oo = np.einsum('ru,rv->ruv', rho_o, rho_o)
            rho_vv = np.einsum('ra,rb->rab', rho_v, rho_v)

            if collinear_samples > 0:
                nimc = dft.numint2c.NumInt2C()
                nimc.collinear = 'mcol'
                nimc.collinear_samples = collinear_samples
                eval_xc_eff_sf = mcfun_eval_xc_adapter_sf(nimc, mf.xc)
                rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                fxc_sf = 2 * eval_xc_eff_sf(mf.xc, rho_z, deriv=2, xctype=xctype)[2]
                wfxc_sf = fxc_sf[0, 0] * weight
                add_sf_xc_terms(rho_cv, rho_co, rho_ov, rho_cc, rho_oo, rho_vv, wfxc_sf)

    elif xctype == 'GGA':
        ao_deriv = 1

        def make_pair_gga(r1, r2):
            r12 = np.einsum('xri,rj->xrij', r1, r2[0])
            r12[1:4] += np.einsum('ri,xrj->xrij', r1[0], r2[1:4])
            return r12

        for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
            rho0a = make_rho(0, ao, mask, xctype)
            rho0b = make_rho(1, ao, mask, xctype)
            if Dz0:
                rho0a = rho0b = 0.5 * (rho0a + rho0b)
            rho_c = lib.einsum('xrp,pi->xri', ao, orbcs)
            rho_o = lib.einsum('xrp,pi->xri', ao, orbos)
            rho_v = lib.einsum('xrp,pi->xri', ao, orbvs)
            rho_cv = make_pair_gga(rho_c, rho_v)
            rho_co = make_pair_gga(rho_c, rho_o)
            rho_ov = make_pair_gga(rho_o, rho_v)
            rho_cc = make_pair_gga(rho_c, rho_c)
            rho_oo = make_pair_gga(rho_o, rho_o)
            rho_vv = make_pair_gga(rho_v, rho_v)

            if collinear_samples > 0:
                nimc = dft.numint2c.NumInt2C()
                nimc.collinear = 'mcol'
                nimc.collinear_samples = collinear_samples
                eval_xc_eff_sf = mcfun_eval_xc_adapter_sf(nimc, mf.xc)
                rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                fxc_sf = 2 * eval_xc_eff_sf(mf.xc, rho_z, deriv=2, xctype=xctype)[2]
                wfxc_sf = fxc_sf * weight
                add_sf_xc_terms(rho_cv, rho_co, rho_ov, rho_cc, rho_oo, rho_vv, wfxc_sf)

    elif xctype == 'MGGA':
        ao_deriv = 1

        def make_pair_mgga(r1, r2):
            r12 = np.einsum('xri,rj->xrij', r1, r2[0])
            r12[1:4] += np.einsum('ri,xrj->xrij', r1[0], r2[1:4])
            tau12 = np.einsum('xri,xrj->rij', r1[1:4], r2[1:4]) * 0.5
            return np.vstack([r12, tau12[np.newaxis]])

        for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
            rho0a = make_rho(0, ao, mask, xctype)
            rho0b = make_rho(1, ao, mask, xctype)
            if Dz0:
                rho0a = rho0b = 0.5 * (rho0a + rho0b)
            rho_c = lib.einsum('xrp,pi->xri', ao, orbcs)
            rho_o = lib.einsum('xrp,pi->xri', ao, orbos)
            rho_v = lib.einsum('xrp,pi->xri', ao, orbvs)
            rho_cv = make_pair_mgga(rho_c, rho_v)
            rho_co = make_pair_mgga(rho_c, rho_o)
            rho_ov = make_pair_mgga(rho_o, rho_v)
            rho_cc = make_pair_mgga(rho_c, rho_c)
            rho_oo = make_pair_mgga(rho_o, rho_o)
            rho_vv = make_pair_mgga(rho_v, rho_v)

            if collinear_samples > 0:
                nimc = dft.numint2c.NumInt2C()
                nimc.collinear = 'mcol'
                nimc.collinear_samples = collinear_samples
                eval_xc_eff_sf = mcfun_eval_xc_adapter_sf(nimc, mf.xc)
                rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                fxc_sf = 2 * eval_xc_eff_sf(mf.xc, rho_z, deriv=2, xctype=xctype)[2]
                wfxc_sf = fxc_sf * weight
                add_sf_xc_terms(rho_cv, rho_co, rho_ov, rho_cc, rho_oo, rho_vv, wfxc_sf)

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'{xctype} not supported for SASF-TDA.')

    fockz_cv = orbcs.T @ fockz @ orbvs
    fockb_co = orbcs.T @ fockb @ orbos
    focka_ov = orbos.T @ focka @ orbvs
    fock0_cv = orbcs.T @ fock0 @ orbvs

    a_cvoo = fockz_cv * 2 * np.sqrt((si + 1) / (2 * si))
    a_cooo = -fockb_co
    a_ooov = focka_ov
    a_cv0oo = -fock0_cv * np.sqrt(2)
    if not with_oo:
        a_cvoo = np.zeros_like(a_cvoo)
        a_cooo = np.zeros_like(a_cooo)
        a_ooov = np.zeros_like(a_ooov)
        a_cv0oo = np.zeros_like(a_cv0oo)

    co = ncs * nos
    cv = ncs * nvs
    ov = nos * nvs
    a = np.zeros((2 * cv + co + ov + 1, 2 * cv + co + ov + 1))

    a[:co, :co] = a_coco.reshape(co, co)
    a[:co, co:co+cv] = a_cvco.transpose(2, 3, 0, 1).reshape(co, cv)
    a[:co, co+cv:co+cv+1] = a_cooo.reshape(co, 1)
    a[:co, co+cv+1:co+cv+ov+1] = a_coov.reshape(co, ov)
    a[:co, co+cv+1+ov:] = a_cv0co.transpose(2, 3, 0, 1).reshape(co, cv)

    a[co:co+cv, :co] = a_cvco.reshape(cv, co)
    a[co:co+cv, co:co+cv] = a_cvcv.reshape(cv, cv)
    a[co:co+cv, co+cv:co+cv+1] = a_cvoo.reshape(cv, 1)
    a[co:co+cv, co+cv+1:co+cv+ov+1] = a_cvov.reshape(cv, ov)
    a[co:co+cv, co+cv+1+ov:] = a_cv0cv.transpose(2, 3, 0, 1).reshape(cv, cv)

    a[co+cv:co+cv+1, :co] = a_cooo.reshape(1, co)
    a[co+cv:co+cv+1, co:co+cv] = a_cvoo.reshape(1, cv)
    a[co+cv:co+cv+1, co+cv:co+cv+1] = 0.0
    a[co+cv:co+cv+1, co+cv+1:co+cv+ov+1] = a_ooov.reshape(1, ov)
    a[co+cv:co+cv+1, co+cv+1+ov:] = a_cv0oo.reshape(1, cv)

    a[co+cv+1:co+cv+ov+1, :co] = a_coov.transpose(2, 3, 0, 1).reshape(ov, co)
    a[co+cv+1:co+cv+ov+1, co:co+cv] = a_cvov.transpose(2, 3, 0, 1).reshape(ov, cv)
    a[co+cv+1:co+cv+ov+1, co+cv:co+cv+1] = a_ooov.reshape(ov, 1)
    a[co+cv+1:co+cv+ov+1, co+cv+1:co+cv+ov+1] = a_ovov.reshape(ov, ov)
    a[co+cv+1:co+cv+ov+1, co+cv+1+ov:] = a_cv0ov.transpose(2, 3, 0, 1).reshape(ov, cv)

    a[co+cv+1+ov:, :co] = a_cv0co.reshape(cv, co)
    a[co+cv+1+ov:, co:co+cv] = a_cv0cv.reshape(cv, cv)
    a[co+cv+1+ov:, co+cv:co+cv+1] = a_cv0oo.reshape(cv, 1)
    a[co+cv+1+ov:, co+cv+1:co+cv+ov+1] = a_cv0ov.reshape(cv, ov)
    a[co+cv+1+ov:, co+cv+1+ov:] = a_cv0cv0.reshape(cv, cv)

    assert abs(a - a.T).max() < 1e-10
    return a


def _sasf_block_slices(nc, no, nv):
    co = nc * no
    cv = nc * nv
    ov = no * nv
    p_co = 0
    p_cv = p_co + co
    p_oo = p_cv + cv
    p_ov = p_oo + 1
    p_cv0 = p_ov + ov
    p_end = p_cv0 + cv
    return {
        'CO(1)': slice(p_co, p_cv),
        'CV(1)': slice(p_cv, p_oo),
        'OO(1)': slice(p_oo, p_ov),
        'OV(1)': slice(p_ov, p_cv0),
        'CV(0)': slice(p_cv0, p_end),
    }


def _sasf_main_component(pos, csidx, osidx, vsidx):
    nc = len(csidx)
    no = len(osidx)
    nv = len(vsidx)
    co = nc * no
    cv = nc * nv
    ov = no * nv

    if pos < co:
        c, o = np.unravel_index(pos, (nc, no))
        return 'CO(1)', csidx[c], osidx[o], c, o

    pos -= co
    if pos < cv:
        c, v = np.unravel_index(pos, (nc, nv))
        return 'CV(1)', csidx[c], vsidx[v], c, v

    pos -= cv
    if pos < 1:
        return 'OO(1)', None, None, None, None

    pos -= 1
    if pos < ov:
        o, v = np.unravel_index(pos, (no, nv))
        return 'OV(1)', osidx[o], vsidx[v], o, v

    pos -= ov
    c, v = np.unravel_index(pos, (nc, nv))
    return 'CV(0)', csidx[c], vsidx[v], c, v


def _sasf_irrep_name(mf, mo1, mo2):
    if mo1 is None or mo2 is None or mf.mol.groupname == 'C1':
        return 'A'

    mol = mf.mol
    orb_sym = mf.get_orbsym(mf.mo_coeff)
    ground_sym = mf.get_wfnsym()
    if isinstance(ground_sym, str):
        ground_sym = mol.irrep_name.index(ground_sym) if ground_sym in mol.irrep_name else 0
    else:
        ground_sym = int(np.asarray(ground_sym).ravel()[0])

    # PySCF's direct_prod can return non-standard IDs for Dooh/Coov components
    # (for example E1u x A1g may not be usable as an index of mol.irrep_name).
    # XOR is the Abelian subgroup product used by PySCF irrep IDs and preserves
    # the concrete x/y component labels printed in the MO table.
    ir = int(orb_sym[mo1]) ^ int(orb_sym[mo2]) ^ ground_sym
    if ir in mol.irrep_id:
        return mol.irrep_name[mol.irrep_id.index(ir)]
    if ir < len(mol.irrep_name):
        return mol.irrep_name[ir]
    return 'A'


def analyze_sasf(e, values, mf, mo_occ=None, threshold=0.1, nstates=None, with_oo=True):
    """Analyze SASF-TDA eigenvectors in the CO(1), CV(1), OO(1), OV(1), CV(0) order."""
    from pyscf.data import nist

    if mo_occ is None:
        mo_occ = mf.mo_occ

    e = np.asarray(e)
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, np.newaxis]

    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]
    nc = len(csidx)
    no = len(osidx)
    nv = len(vsidx)

    slices = _sasf_block_slices(nc, no, nv)
    ndim = slices['CV(0)'].stop
    if values.shape[0] != ndim:
        raise ValueError(f'values.shape[0] = {values.shape[0]}, expected {ndim}')

    if nstates is None:
        nstates = min(len(e), values.shape[1])
    else:
        nstates = min(nstates, len(e), values.shape[1])

    syms = []
    for istate in range(nstates):
        value = values[:, istate]
        x_co1 = value[slices['CO(1)']].reshape(nc, no)
        x_cv1 = value[slices['CV(1)']].reshape(nc, nv)
        x_oo1 = value[slices['OO(1)']]
        x_ov1 = value[slices['OV(1)']].reshape(no, nv)
        x_cv0 = value[slices['CV(0)']].reshape(nc, nv)

        block, mo1, mo2, _, _ = _sasf_main_component(int(np.argmax(value**2)), csidx, osidx, vsidx)
        sym = _sasf_irrep_name(mf, mo1, mo2)
        syms.append(sym)

        note = ''
        if block == 'OO(1)' and not with_oo:
            note = ' (zero OO mode from with_oo=False)'
        print(f'Excited state {istate + 1} {e[istate] * nist.HARTREE2EV:12.5f} eV, symmetry={sym}{note}')

        for c, o in zip(*np.where(abs(x_co1) > threshold)):
            print(f'CO(1) {csidx[c] + 1} -> {osidx[o] + 1} {x_co1[c, o]:10.5f} '
                  f'{100 * x_co1[c, o] ** 2:5.2f}%')
        for c, v in zip(*np.where(abs(x_cv1) > threshold)):
            print(f'CV(1) {csidx[c] + 1} -> {vsidx[v] + 1} {x_cv1[c, v]:10.5f} '
                  f'{100 * x_cv1[c, v] ** 2:5.2f}%')
        if abs(x_oo1[0]) > threshold:
            oo_note = ' [zeroed]' if not with_oo else ''
            print(f'OO(1){oo_note} {x_oo1[0]:10.5f} {100 * x_oo1[0] ** 2:5.2f}%')
        for o, v in zip(*np.where(abs(x_ov1) > threshold)):
            print(f'OV(1) {osidx[o] + 1} -> {vsidx[v] + 1} {x_ov1[o, v]:10.5f} '
                  f'{100 * x_ov1[o, v] ** 2:5.2f}%')
        for c, v in zip(*np.where(abs(x_cv0) > threshold)):
            print(f'CV(0) {csidx[c] + 1} -> {vsidx[v] + 1} {x_cv0[c, v]:10.5f} '
                  f'{100 * x_cv0[c, v] ** 2:5.2f}%')
        print(' ')

    eo = e[:nstates] * nist.HARTREE2EV
    print('=' * 60)
    print('SASF-TDA analysis')
    for istate in range(nstates):
        print(f'No.{istate:3d}  Esf={eo[istate]:>10.5f} eV,  '
              f'En-E1={(eo[istate] - eo[0]):>10.5f} eV,  symmetry={syms[istate]}')
    return syms

