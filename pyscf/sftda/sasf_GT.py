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
# Generalized triplet SASF-TDA matrix with spin-flip kernels K^SF.
#

import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf import dft
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.tdscf.uhf import TDBase
from pyscf import __config__
from pyscf.data import nist

from pyscf.sftda.numint2c_sftd import mcfun_eval_xc_adapter_sf


MO_BASE = getattr(__config__, 'MO_BASE', 1)


def _pair_density(ao, orb1, orb2, xctype):
    if xctype == 'LDA':
        rho1 = lib.einsum('rp,pi->ri', ao, orb1)
        rho2 = lib.einsum('rp,pi->ri', ao, orb2)
        return np.einsum('ri,rj->rij', rho1, rho2)

    rho1 = lib.einsum('xrp,pi->xri', ao, orb1)
    rho2 = lib.einsum('xrp,pi->xri', ao, orb2)
    rho12 = np.einsum('xri,rj->xrij', rho1, rho2[0])
    rho12[1:4] += np.einsum('ri,xrj->xrij', rho1[0], rho2[1:4])

    if xctype == 'MGGA':
        tau12 = np.einsum('xri,xrj->rij', rho1[1:4], rho2[1:4]) * 0.5
        rho12 = np.vstack([rho12, tau12[np.newaxis]])
    return rho12


def _contract_sf_kernel(rho_l, rho_r, wfxc, xctype):
    if xctype == 'LDA':
        w_r = np.einsum('gij,g->gij', rho_r, wfxc)
        return lib.einsum('gij,gkl->ijkl', rho_l, w_r)

    w_r = np.einsum('xyg,ygij->xgij', wfxc, rho_r)
    return lib.einsum('xgij,xgkl->ijkl', rho_l, w_r)


def _add_sf_exchange_(mol, kernels, orbmap, hyb=0.0, omega=0.0, alpha=0.0):
    def exchange(p, q, r, s, coeff):
        if coeff == 0:
            return np.zeros((
                orbmap[p].shape[1], orbmap[q].shape[1],
                orbmap[r].shape[1], orbmap[s].shape[1],
            ))

        eri = ao2mo.general(
            mol, [orbmap[q], orbmap[s], orbmap[r], orbmap[p]], compact=False
        )
        eri = eri.reshape(
            orbmap[q].shape[1], orbmap[s].shape[1],
            orbmap[r].shape[1], orbmap[p].shape[1],
        )
        return -coeff * eri.transpose(3, 0, 2, 1)

    def add_one(key, p, q, r, s, coeff):
        kernels[key] += exchange(p, q, r, s, coeff)

    if hyb != 0:
        for key, p, q, r, s in _SF_KERNEL_SPECS:
            add_one(key, p, q, r, s, hyb)

    k_fac = alpha - hyb
    if omega != 0 and k_fac != 0:
        with mol.with_range_coulomb(omega):
            for key, p, q, r, s in _SF_KERNEL_SPECS:
                add_one(key, p, q, r, s, k_fac)


_SF_KERNEL_SPECS = (
    ('ai_uj', 'v', 'c', 'o', 'c'),
    ('ai_bu', 'v', 'c', 'v', 'o'),
    ('ui_vj', 'o', 'c', 'o', 'c'),
    ('uv_ij', 'o', 'o', 'c', 'c'),
    ('ub_iv', 'o', 'v', 'c', 'o'),
    ('au_bv', 'v', 'o', 'v', 'o'),
    ('ab_uv', 'v', 'v', 'o', 'o'),
    ('ai_bj', 'v', 'c', 'v', 'c'),
    ('ab_ij', 'v', 'v', 'c', 'c'),
    ('au_ij', 'v', 'o', 'c', 'c'),
    ('ab_iu', 'v', 'v', 'c', 'o'),
)


def _make_empty_sf_kernels(ncs, nos, nvs):
    dims = {'c': ncs, 'o': nos, 'v': nvs}
    return {
        key: np.zeros((dims[p], dims[q], dims[r], dims[s]))
        for key, p, q, r, s in _SF_KERNEL_SPECS
    }


def _build_sf_kernels(mf, orbcs, orbos, orbvs, collinear_samples):
    mol = mf.mol
    ncs = orbcs.shape[1]
    nos = orbos.shape[1]
    nvs = orbvs.shape[1]
    nao = orbcs.shape[0]
    kernels = _make_empty_sf_kernels(ncs, nos, nvs)

    orbmap = {'c': orbcs, 'o': orbos, 'v': orbvs}
    omega = alpha = 0.0
    hyb = 1.0
    xctype = 'HF'

    if isinstance(mf, dft.KohnShamDFT):
        ni = mf._numint
        xctype = ni._xc_type(mf.xc)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        if collinear_samples > 0 and xctype != 'HF':
            ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
            dm0 = mf.to_uks().make_rdm1()
            make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1, with_lapl=False)[0]
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory * 0.8 - mem_now)

            nimc = dft.numint2c.NumInt2C()
            nimc.collinear = 'mcol'
            nimc.collinear_samples = collinear_samples
            eval_xc_eff_sf = mcfun_eval_xc_adapter_sf(nimc, mf.xc)

            ao_deriv = 0 if xctype == 'LDA' else 1
            if xctype not in ('LDA', 'GGA', 'MGGA'):
                raise NotImplementedError(
                    'Only LDA/GGA/MGGA/HF are implemented for sasf_GT.'
                )

            for ao, mask, weight, coords in ni.block_loop(
                    mol, mf.grids, nao, ao_deriv, max_memory):
                rho0a = make_rho(0, ao, mask, xctype)
                rho0b = make_rho(1, ao, mask, xctype)
                rho_z = np.array([rho0a + rho0b, rho0a - rho0b])
                fxc_sf = eval_xc_eff_sf(mf.xc, rho_z, deriv=2, xctype=xctype)[2]

                if xctype == 'LDA':
                    wfxc_sf = 2.0 * fxc_sf[0, 0] * weight
                else:
                    wfxc_sf = 2.0 * fxc_sf * weight

                rho_vc = _pair_density(ao, orbvs, orbcs, xctype)
                rho_oc = _pair_density(ao, orbos, orbcs, xctype)
                rho_co = _pair_density(ao, orbcs, orbos, xctype)
                rho_ov = _pair_density(ao, orbos, orbvs, xctype)
                rho_vo = _pair_density(ao, orbvs, orbos, xctype)
                rho_oo = _pair_density(ao, orbos, orbos, xctype)
                rho_cc = _pair_density(ao, orbcs, orbcs, xctype)
                rho_vv = _pair_density(ao, orbvs, orbvs, xctype)

                kernels['ai_uj'] += _contract_sf_kernel(rho_vc, rho_oc, wfxc_sf, xctype)
                kernels['ai_bu'] += _contract_sf_kernel(rho_vc, rho_vo, wfxc_sf, xctype)
                kernels['ui_vj'] += _contract_sf_kernel(rho_oc, rho_oc, wfxc_sf, xctype)
                kernels['uv_ij'] += _contract_sf_kernel(rho_oo, rho_cc, wfxc_sf, xctype)
                kernels['ub_iv'] += _contract_sf_kernel(rho_ov, rho_co, wfxc_sf, xctype)
                kernels['au_bv'] += _contract_sf_kernel(rho_vo, rho_vo, wfxc_sf, xctype)
                kernels['ab_uv'] += _contract_sf_kernel(rho_vv, rho_oo, wfxc_sf, xctype)
                kernels['ai_bj'] += _contract_sf_kernel(rho_vc, rho_vc, wfxc_sf, xctype)
                kernels['ab_ij'] += _contract_sf_kernel(rho_vv, rho_cc, wfxc_sf, xctype)
                kernels['au_ij'] += _contract_sf_kernel(rho_vo, rho_cc, wfxc_sf, xctype)
                kernels['ab_iu'] += _contract_sf_kernel(rho_vv, rho_co, wfxc_sf, xctype)
    else:
        omega = alpha = 0.0
        hyb = 1.0

    _add_sf_exchange_(mol, kernels, orbmap, hyb=hyb, omega=omega, alpha=alpha)
    return kernels


def _basis_info(ncs, nos, nvs):
    sizes = {
        'OO': nos * nos,
        'CO': ncs * nos,
        'CV': ncs * nvs,
        'CVN': ncs * nvs,
        'OV': nos * nvs,
    }

    slices = {}
    p0 = 0
    for key in ('OO', 'CO', 'CV', 'CVN', 'OV'):
        p1 = p0 + sizes[key]
        slices[key] = slice(p0, p1)
        p0 = p1

    return {
        'slices': slices,
        'sizes': sizes,
        'shapes': {
            'OO': (nos, nos),
            'CO': (ncs, nos),
            'CV': (ncs, nvs),
            'CVN': (ncs, nvs),
            'OV': (nos, nvs),
        },
        'ncs': ncs,
        'nos': nos,
        'nvs': nvs,
        'size': p0,
    }


def _oo_vred(info):
    nos = info['nos']
    vred = np.zeros(info['shapes']['OO'])
    if nos > 0:
        np.fill_diagonal(vred, 1.0 / np.sqrt(nos))
    return vred


def _oo_projector(info):
    return _oo_vred(info).reshape(-1)


def amplitudes_to_blocks(vector, info):
    vector = np.asarray(vector)
    slices = info['slices']
    shapes = info['shapes']
    return {
        key: vector[slices[key]].reshape(shapes[key])
        for key in ('OO', 'CO', 'CV', 'CVN', 'OV')
    }


def _state_metrics(vector, info):
    blocks = amplitudes_to_blocks(vector, info)
    norm2 = np.vdot(vector, vector).real
    if norm2 < 1e-24:
        norm2 = 1.0

    weights = {
        key: np.vdot(blocks[key], blocks[key]).real / norm2
        for key in ('OO', 'CO', 'CV', 'CVN', 'OV')
    }
    oo_norm = np.sqrt(max(np.vdot(blocks['OO'], blocks['OO']).real, 0.0))
    if oo_norm > 1e-12:
        vred_overlap = abs(np.vdot(_oo_vred(info), blocks['OO']) / oo_norm)
    else:
        vred_overlap = 0.0
    total_vred_overlap = abs(np.vdot(_oo_vred(info), blocks['OO']) / np.sqrt(norm2))
    return weights, float(vred_overlap), float(total_vred_overlap)


def _classify_oo_states(vectors, info, oo_threshold=0.5):
    nstates = len(vectors)
    oo_weights = np.zeros(nstates)
    vred_overlaps = np.zeros(nstates)
    total_vred_overlaps = np.zeros(nstates)

    for i, vec in enumerate(vectors):
        weights, vred_overlap, total_vred_overlap = _state_metrics(vec, info)
        oo_weights[i] = weights['OO']
        vred_overlaps[i] = vred_overlap
        total_vred_overlaps[i] = total_vred_overlap

    if nstates == 0:
        special_idx = None
    else:
        special_idx = int(np.argmax(total_vred_overlaps))

    oo_dominant = oo_weights >= oo_threshold
    if special_idx is not None:
        oo_dominant[special_idx] = True

    remove_mask = oo_dominant.copy()
    if special_idx is not None:
        remove_mask[special_idx] = False

    return {
        'oo_weights': oo_weights,
        'vred_overlaps': vred_overlaps,
        'total_vred_overlaps': total_vred_overlaps,
        'oo_dominant': oo_dominant,
        'remove_mask': remove_mask,
        'special_idx': special_idx,
        'expected_remove': max(info['sizes']['OO'] - 1, 0),
    }


def _format_percent(x):
    return x * 100.0


def _iter_large_amplitudes(vector, info, threshold=0.1):
    blocks = amplitudes_to_blocks(vector, info)
    ncs = info['ncs']
    nos = info['nos']
    for key in ('OO', 'CO', 'CV', 'CVN', 'OV'):
        block = blocks[key]
        for idx in zip(*np.where(abs(block) > threshold)):
            amp = block[idx]
            if key == 'OO':
                u, v = idx
                alpha_idx = ncs + u
                beta_idx = ncs + v
                label = 'OO'
            elif key == 'CO':
                i, u = idx
                alpha_idx = i
                beta_idx = ncs + u
                label = 'CO'
            elif key in ('CV', 'CVN'):
                i, a = idx
                alpha_idx = i
                beta_idx = ncs + nos + a
                label = 'CV(N)' if key == 'CVN' else 'CV'
            else:
                u, a = idx
                alpha_idx = ncs + u
                beta_idx = ncs + nos + a
                label = 'OV'
            yield label, alpha_idx, beta_idx, amp


def analyze(tdobj, verbose=None):
    log = logger.new_logger(tdobj, verbose)
    info = getattr(tdobj, 'gt_basis', None)
    if info is None:
        _, info = tdobj.get_a_sasf(return_info=True)

    vectors = [xy[0] for xy in tdobj.xy]
    metrics = _classify_oo_states(
        vectors, info, getattr(tdobj, 'oo_dominance_threshold', 0.5))
    full_info = getattr(tdobj, 'gt_state_info', None)
    final_to_full = getattr(tdobj, 'gt_final_to_full', np.arange(len(vectors)))

    if getattr(tdobj, 'remove', False):
        removed = getattr(tdobj, 'gt_removed_oo_states', [])
        if removed:
            msg = ', '.join(
                'State %d (E=%.5f eV, OO=%.1f%%, v_red=%.4f)' %
                (item['state'] + 1, item['energy_ev'],
                 _format_percent(item['oo_weight']), item['vred_overlap'])
                for item in removed
            )
            log.note('Removed OO-space companion states: %s', msg)

    e_ev_array = np.asarray(tdobj.e) * nist.HARTREE2EV
    nstates = min(tdobj.nstates, len(tdobj.xy))

    for i in range(nstates):
        x = vectors[i]
        e_ev = e_ev_array[i]
        weights, vred_overlap, total_vred_overlap = _state_metrics(x, info)

        if full_info is not None and i < len(final_to_full):
            full_idx = int(final_to_full[i])
            is_special = full_idx == full_info['special_idx']
            is_oo_dominant = bool(full_info['oo_dominant'][full_idx])
        else:
            is_special = i == metrics['special_idx']
            is_oo_dominant = bool(metrics['oo_dominant'][i])

        if is_special:
            tag = 'OO reference v_red adopted'
        elif is_oo_dominant:
            tag = 'OO-space companion'
        else:
            tag = 'generalized triplet'

        log.note(
            'Excited State %3d: %12.5f eV  [%s; '
            'OO=%.2f%%, v_red=%.4f, CO=%.2f%%, CV=%.2f%%, '
            'CV(N)=%.2f%%, OV=%.2f%%]',
            i + 1, e_ev, tag,
            _format_percent(weights['OO']), vred_overlap,
            _format_percent(weights['CO']), _format_percent(weights['CV']),
            _format_percent(weights['CVN']), _format_percent(weights['OV']),
        )

        if log.verbose >= logger.INFO:
            norm = np.vdot(x, x).real
            if norm < 1e-12:
                norm = 1.0
            for label, alpha_idx, beta_idx, amp in _iter_large_amplitudes(x, info):
                weight = abs(amp)**2 / norm * 100.0
                log.info(' %2d%% %s(ab) %da -> %db    %.5f',
                         int(round(weight)), label,
                         alpha_idx + MO_BASE, beta_idx + MO_BASE, amp)


def get_h_sasf_gt(mf, mo_energy=None, mo_coeff=None, mo_occ=None,
                  collinear_samples=20, return_info=False):
    r'''
    Build the generalized-triplet SASF-TDA matrix.

    The basis order is ``OO, CO, CV, CVN, OV``.  ``CV`` and ``CVN`` are two
    independent bases for each closed-shell-to-virtual excitation.
    '''
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mol = mf.mol
    si = (mol.nelec[0] - mol.nelec[1]) * 0.5
    if si <= 0:
        raise ValueError('The generalized-triplet SASF matrix requires S_i > 0.')

    csidx = np.where(mo_occ == 2)[0]
    osidx = np.where(mo_occ == 1)[0]
    vsidx = np.where(mo_occ == 0)[0]
    orbcs = mo_coeff[:, csidx]
    orbos = mo_coeff[:, osidx]
    orbvs = mo_coeff[:, vsidx]
    ncs = orbcs.shape[1]
    nos = orbos.shape[1]
    nvs = orbvs.shape[1]

    info = _basis_info(ncs, nos, nvs)
    slices = info['slices']
    h = np.zeros((info['size'], info['size']))

    def put(row, col, block, hermi=True):
        srow = slices[row]
        scol = slices[col]
        block = np.asarray(block).reshape(srow.stop - srow.start,
                                          scol.stop - scol.start)
        h[srow, scol] += block
        if hermi and row != col:
            h[scol, srow] += block.conj().T

    fock = mf.get_fock()
    focka = fock.focka
    fockb = fock.fockb
    fock0 = 0.5 * (focka + fockb)
    fockz = 0.5 * (focka - fockb)

    orbcs.conj().T @ focka @ orbcs
    fa_oc = orbos.conj().T @ focka @ orbcs
    fa_oo = orbos.conj().T @ focka @ orbos
    fa_ov = orbos.conj().T @ focka @ orbvs
    fa_vv = orbvs.conj().T @ focka @ orbvs

    fb_cc = orbcs.conj().T @ fockb @ orbcs
    fb_co = orbcs.conj().T @ fockb @ orbos
    fb_oo = orbos.conj().T @ fockb @ orbos
    fb_vo = orbvs.conj().T @ fockb @ orbos

    f0_cc = orbcs.conj().T @ fock0 @ orbcs
    f0_vc = orbvs.conj().T @ fock0 @ orbcs
    f0_vv = orbvs.conj().T @ fock0 @ orbvs

    fz_cv = orbcs.conj().T @ fockz @ orbvs
    fz_cc = orbcs.conj().T @ fockz @ orbcs
    fz_vv = orbvs.conj().T @ fockz @ orbvs

    eye_c = np.eye(ncs)
    eye_o = np.eye(nos)
    eye_v = np.eye(nvs)

    kernels = _build_sf_kernels(mf, orbcs, orbos, orbvs, collinear_samples)

    pref_cv = np.sqrt((si + 1.0) / (2.0 * si))
    pref_oo_cv = 2.0 * pref_cv
    pref_cvn_cv = -np.sqrt((si + 1.0) / si)

    oo_ref = _oo_projector(info)
    put('OO', 'CV', np.einsum('x,ia->xia', oo_ref, pref_oo_cv * fz_cv))
    put('OO', 'CO', np.einsum('x,iu->xiu', oo_ref, -fb_co))
    put('OO', 'OV', np.einsum('x,ua->xua', oo_ref, fa_ov))

    h_cv_co = pref_cv * (
        np.einsum('ij,au->iaju', eye_c, fb_vo) +
        kernels['ai_uj'].transpose(1, 0, 3, 2)
    )
    put('CV', 'CO', h_cv_co)

    h_cv_ov = pref_cv * (
        -np.einsum('ab,ui->iaub', eye_v, fa_oc) +
        kernels['ai_bu'].transpose(1, 0, 3, 2)
    )
    put('CV', 'OV', h_cv_ov)

    h_co_co = (
        np.einsum('ij,uv->iujv', eye_c, fb_oo) -
        np.einsum('uv,ji->iujv', eye_o, fb_cc) +
        kernels['ui_vj'].transpose(1, 0, 3, 2) -
        kernels['uv_ij'].transpose(2, 0, 3, 1)
    )
    put('CO', 'CO', h_co_co, hermi=False)

    h_co_ov = kernels['ub_iv'].transpose(2, 0, 3, 1)
    put('CO', 'OV', h_co_ov)

    h_ov_ov = (
        np.einsum('uv,ab->uavb', eye_o, fa_vv) -
        np.einsum('ab,vu->uavb', eye_v, fa_oo) +
        kernels['au_bv'].transpose(1, 0, 3, 2) -
        kernels['ab_uv'].transpose(2, 0, 3, 1)
    )
    put('OV', 'OV', h_ov_ov, hermi=False)

    h_cv_cv = (
        kernels['ai_bj'].transpose(1, 0, 3, 2) +
        np.einsum('ij,ab->iajb', eye_c, f0_vv - fz_vv / si) -
        np.einsum('ab,ji->iajb', eye_v, f0_cc + fz_cc / si)
    )
    put('CV', 'CV', h_cv_cv, hermi=False)

    h_cvn_cvn = (
        np.einsum('ij,ab->iajb', eye_c, f0_vv) -
        np.einsum('ab,ij->iajb', eye_v, f0_cc) +
        kernels['ai_bj'].transpose(1, 0, 3, 2) -
        2.0 * kernels['ab_ij'].transpose(2, 0, 3, 1)
    )
    put('CVN', 'CVN', h_cvn_cvn, hermi=False)

    put('CVN', 'OO', np.einsum('ia,x->iax', -np.sqrt(2.0) * f0_vc.T, oo_ref))

    h_cvn_co = (
        np.einsum('ij,au->iaju', eye_c, fb_vo) +
        kernels['ai_uj'].transpose(1, 0, 3, 2) -
        2.0 * kernels['au_ij'].transpose(2, 0, 3, 1)
    ) / np.sqrt(2.0)
    put('CVN', 'CO', h_cvn_co)

    h_cvn_ov = (
        np.einsum('ab,ui->iaub', eye_v, fa_oc) -
        kernels['ai_bu'].transpose(1, 0, 3, 2) +
        2.0 * kernels['ab_iu'].transpose(2, 0, 3, 1)
    ) / np.sqrt(2.0)
    put('CVN', 'OV', h_cvn_ov)

    h_cvn_cv = pref_cvn_cv * (
        np.einsum('ij,ab->iajb', eye_c, fz_vv) -
        np.einsum('ab,ji->iajb', eye_v, fz_cc)
    )
    put('CVN', 'CV', h_cvn_cv)

    if return_info:
        return h, info
    return h


def get_a_sasf(mf, mo_energy=None, mo_coeff=None, mo_occ=None,
               collinear_samples=20, return_info=False):
    return get_h_sasf_gt(
        mf, mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ,
        collinear_samples=collinear_samples, return_info=return_info)


class TDA_SASF(TDBase):
    collinear_samples = getattr(
        __config__, 'tdscf_uhf_sf_SF-TDA_collinear_samples', 20)
    remove = False
    oo_dominance_threshold = 0.5
    _keys = {
        'collinear_samples', 'gt_basis', 'extype', 'remove',
        'oo_dominance_threshold', 'gt_state_info', 'gt_final_to_full',
        'gt_removed_oo_states',
    }

    def __init__(self, mf, collinear_samples=20, remove=False,
                 oo_dominance_threshold=0.5):
        TDBase.__init__(self, mf)
        self.collinear_samples = collinear_samples
        self.gt_basis = None
        self.remove = remove
        self.oo_dominance_threshold = oo_dominance_threshold
        self.gt_state_info = None
        self.gt_final_to_full = None
        self.gt_removed_oo_states = []
        self.extype = getattr(self, 'extype', 1)
        if self.extype != 1:
            raise NotImplementedError("Only spin flip down is allowed.")

    def get_a_sasf(self, mf=None, collinear_samples=None, return_info=False):
        if mf is None:
            mf = self._scf
        if collinear_samples is None:
            collinear_samples = self.collinear_samples
        return get_a_sasf(
            mf, collinear_samples=collinear_samples, return_info=return_info)

    def kernel(self, x0=None, nstates=None, extype=None):
        cpu0 = (logger.process_clock(), logger.perf_counter())

        self.check_sanity()
        self.dump_flags()

        if extype is not None and extype != 1:
            raise NotImplementedError("Only spin flip down is allowed.")

        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)

        h, info = self.get_a_sasf(return_info=True)
        norm = np.linalg.norm(h)
        antihermi = np.linalg.norm(h - h.conj().T)
        if antihermi > 1e-8 * max(norm, 1.0):
            log.warn('Generalized-triplet matrix is not Hermitian: '
                     '|H-H^H| = %.6e, |H| = %.6e. '
                     'Using the Hermitian part for full diagonalization.',
                     antihermi, norm)
        h = (h + h.conj().T) * 0.5

        e, x = np.linalg.eigh(h)
        idx = np.argsort(e)
        e = e[idx]
        x = x[:, idx].T

        classify = _classify_oo_states(x, info, self.oo_dominance_threshold)
        expected_remove = classify['expected_remove']
        actual_remove = int(np.count_nonzero(classify['remove_mask']))
        special_idx = classify['special_idx']

        if special_idx is not None:
            log.note('OO reference v_red identified as State %d '
                     '(OO=%.2f%%, v_red=%.4f, E=%.5f eV)',
                     special_idx + 1,
                     _format_percent(classify['oo_weights'][special_idx]),
                     classify['vred_overlaps'][special_idx],
                     e[special_idx] * nist.HARTREE2EV)

        oo_count = int(np.count_nonzero(classify['oo_dominant']))
        expected_oo = info['sizes']['OO']
        if oo_count != expected_oo:
            log.warn('Identified %d OO-dominant states, expected %d from '
                     '(2S_i)^2. You may need to adjust '
                     'oo_dominance_threshold=%.3f.',
                     oo_count, expected_oo, self.oo_dominance_threshold)

        keep_all = np.arange(e.size)
        removed = []
        if self.remove:
            remove_idx = np.where(classify['remove_mask'])[0]
            if actual_remove != expected_remove:
                log.warn('Removing %d OO-space companion states, expected %d '
                         'from (2S_i)^2 - 1.',
                         actual_remove, expected_remove)

            for j in remove_idx:
                removed.append({
                    'state': int(j),
                    'energy_ev': float(e[j] * nist.HARTREE2EV),
                    'oo_weight': float(classify['oo_weights'][j]),
                    'vred_overlap': float(classify['vred_overlaps'][j]),
                })
            if removed:
                log.note('Remove %d OO-space companion states; keep State %d '
                         'as the v_red generalized-triplet reference.',
                         len(removed), special_idx + 1)
                for item in removed:
                    log.info(' remove State %d (OO=%.2f%%, v_red=%.4f, '
                             'E=%.5f eV)',
                             item['state'] + 1,
                             _format_percent(item['oo_weight']),
                             item['vred_overlap'], item['energy_ev'])

            keep_all = np.delete(keep_all, remove_idx)

        keep = keep_all[:min(nstates, keep_all.size)]
        if keep.size < nstates:
            log.warn('Requested %d states, but the generalized-triplet '
                     'matrix only provides %d states.', nstates, keep.size)

        self.e = e[keep]
        self.xy = [(x[i], 0) for i in keep]
        self.converged = np.ones(len(self.e), dtype=bool)
        self.gt_basis = info
        self.gt_state_info = classify
        self.gt_final_to_full = keep
        self.gt_removed_oo_states = removed

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('Generalized-triplet SASF-TDA full diagonalization', *cpu0)
        self._finalize()
        return self.e, self.xy

    analyze = analyze


if __name__ == '__main__':
    mol = gto.M(atom='O; O 1 1.2', charge=0, spin=2, basis='ccpvdz')
    mf = mol.ROKS(xc='B3LYP')
    mf.kernel()
    td = TDA_SASF(mf, collinear_samples=20)
    td.nstates = 5
    td.verbose = 9
    td.kernel()
