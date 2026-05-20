import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf import dft
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.tdscf.uhf import TDBase
from pyscf import __config__
from pyscf.tdscf._lr_eig import eigh as lr_eigh
from pyscf.sftda.sasf import nr_rks_fxc1_gga, nr_rks_fxc1_mgga

def gen_rohf_response_sf(mf, mo_coeff=None, mo_occ=None, hermi=0, max_memory=None):
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

class SATDA(TDBase):

    def gen_vind(self):
        mf = self._scf
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

        vresp, fockz = gen_rohf_response_sf(mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0, max_memory=self.max_memory)

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
    
    def init_guess(self, hdiag, nstates=None):
        if nstates is None:
            nstates = self.nstates
        n_init = min(nstates + 3, hdiag.size)
        idx = np.argsort(hdiag)[:n_init]
        x0 = np.zeros((n_init, hdiag.size))
        x0[np.arange(n_init), idx] = 1.0
        return x0

    def kernel(self, x0=None, nstates=None, extype=None):
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

        nocca, noccb = self.mol.nelec
        nvirb = self._scf.mo_occ.size - noccb

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

        self.xy = [(xi.reshape(nocca, nvirb), 0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('SATDA', *cpu0)
        self._finalize()
        return self.e, self.xy
