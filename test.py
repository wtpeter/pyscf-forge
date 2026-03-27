#!/usr/bin/env python

import importlib.util
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
PYSCF_FORGE_PATH = ROOT / 'pyscf'
if str(PYSCF_FORGE_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyscf

if str(PYSCF_FORGE_PATH) in pyscf.__path__:
    pyscf.__path__.remove(str(PYSCF_FORGE_PATH))
pyscf.__path__.insert(0, str(PYSCF_FORGE_PATH))

from pyscf import dft
from pyscf import gto
from pyscf import sftda
from pyscf.sftda.uhf_sf import get_ab_sf


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


packaged_nac = load_module('packaged_nac_local', ROOT / 'pyscf' / 'nac' / 'tduks_sf.py')


def _solve_manual_tddft(mf, extype=1, collinear_samples=50):
    a, b = get_ab_sf(mf, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b

    mo_occ = mf.mo_occ
    n_occ_a = int((mo_occ[0] > 0).sum())
    n_virt_a = int((mo_occ[0] == 0).sum())
    n_occ_b = int((mo_occ[1] > 0).sum())
    n_virt_b = int((mo_occ[1] == 0).sum())

    A_abab_2d = A_abab.reshape((n_occ_a * n_virt_b, n_occ_a * n_virt_b))
    B_abba_2d = B_abba.reshape((n_occ_a * n_virt_b, n_occ_b * n_virt_a))
    B_baab_2d = B_baab.reshape((n_occ_b * n_virt_a, n_occ_a * n_virt_b))
    A_baba_2d = A_baba.reshape((n_occ_b * n_virt_a, n_occ_b * n_virt_a))

    casida_matrix = np.block([
        [A_abab_2d, B_abba_2d],
        [-B_baab_2d, -A_baba_2d],
    ])

    eigenvals, eigenvecs = np.linalg.eig(casida_matrix)
    idx = eigenvals.real.argsort()
    eigenvals = eigenvals[idx].real
    eigenvecs = eigenvecs[:, idx]

    norms = np.linalg.norm(eigenvecs[:n_occ_a * n_virt_b], axis=0) ** 2
    norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b:], axis=0) ** 2

    if extype == 1:
        valid_mask = norms > 0
    else:
        valid_mask = norms < 0
    sf_eigenvals = eigenvals[valid_mask]
    sf_eigenvecs = eigenvecs[:, valid_mask].T

    return sf_eigenvals, sf_eigenvecs, n_occ_a, n_virt_a, n_occ_b, n_virt_b


def build_manual_tddft_object(
    mf,
    extype=1,
    collinear_samples=50,
    xy_format='old',
    solved_data=None,
):
    if solved_data is None:
        solved_data = _solve_manual_tddft(mf, extype=extype, collinear_samples=collinear_samples)
    sf_eigenvals, sf_eigenvecs, n_occ_a, n_virt_a, n_occ_b, n_virt_b = solved_data

    def norm_xy_old(z):
        x_flat = z[:n_occ_a * n_virt_b]
        y_flat = z[n_occ_a * n_virt_b:]
        norm_val = np.linalg.norm(x_flat) ** 2 - np.linalg.norm(y_flat) ** 2
        norm_val = np.sqrt(1.0 / norm_val)
        x = x_flat.reshape(n_occ_a, n_virt_b) * norm_val
        y = y_flat.reshape(n_occ_b, n_virt_a) * norm_val
        return ((0, x), (y, 0)) if extype == 1 else ((y, 0), (0, x))

    def norm_xy_new(z):
        x_flat = z[:n_occ_a * n_virt_b]
        y_flat = z[n_occ_a * n_virt_b:]
        norm_val = np.linalg.norm(x_flat) ** 2 - np.linalg.norm(y_flat) ** 2
        norm_val = np.sqrt(1.0 / norm_val)
        x = x_flat.reshape(n_occ_a, n_virt_b) * norm_val
        y = y_flat.reshape(n_occ_b, n_virt_a) * norm_val
        return (x, y) if extype == 1 else (y, x)

    td = sftda.uks_sf.TDDFT_SF(mf)
    td.e = sf_eigenvals
    if xy_format == 'old':
        td.xy = [norm_xy_old(z) for z in sf_eigenvecs]
    elif xy_format == 'new':
        td.xy = [norm_xy_new(z) for z in sf_eigenvecs]
    else:
        raise ValueError(f'Unsupported xy_format: {xy_format}')
    td.nstates = len(td.e)
    td.extype = extype
    td.collinear_samples = collinear_samples
    return td


def build_forge_tddft_object(mf, extype=1, collinear_samples=50, nstates=4):
    td = mf.TDDFT_SF()
    td.extype = extype
    td.collinear_samples = collinear_samples
    td.nstates = nstates
    td.kernel()
    return td


def run_case(xc, use_etfs, extype=1, collinear_samples=50, state_i=1, state_j=3):
    cache_module = load_module(f'tduks_sf_cache_ref_{xc}_{use_etfs}', ROOT.parent / 'tduks_sf_cache.py')
    mol = gto.Mole()
    mol.atom = '''
C      0.000000    0.000000    0.000000
O      0.000000    0.000000    1.205000
H     -0.937704    0.000000   -0.513544
H      0.937704    0.100000   -0.513544
'''
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.verbose = 0
    mol.output = '/dev/null'
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = xc
    mf.kernel()

    solved_data = _solve_manual_tddft(mf, extype=extype, collinear_samples=collinear_samples)
    td_old = build_manual_tddft_object(
        mf, extype=extype, collinear_samples=collinear_samples, xy_format='old', solved_data=solved_data
    )
    td_new = build_manual_tddft_object(
        mf, extype=extype, collinear_samples=collinear_samples, xy_format='new', solved_data=solved_data
    )

    baseline = cache_module.NAC(td_old)
    baseline.state_I = state_i
    baseline.state_J = state_j
    baseline.use_etfs = use_etfs
    baseline.ediff = False
    ref = baseline.kernel()

    candidate = packaged_nac.NAC(td_new)
    candidate.state_I = state_i
    candidate.state_J = state_j
    candidate.use_etfs = use_etfs
    candidate.ediff = False
    val = candidate.kernel()

    diff = np.max(np.abs(ref - val))
    print(f'XC={xc} use_etfs={use_etfs} max_abs_diff={diff:.12e}')
    print('Reference NAC (cache):')
    print(ref)
    print('Packaged NAC (new):')
    print(val)

    np.testing.assert_allclose(val, ref, rtol=1e-7, atol=1e-8)


def run_forge_format_smoke_case(xc, use_etfs, extype=1, collinear_samples=50, state_i=1, state_j=2):
    mol = gto.Mole()
    mol.atom = '''
O     0.000000    0.000000    0.000000
H     0.000000   -0.757000    0.587000
H     0.000000    0.757000    0.587000
'''
    mol.basis = '631g'
    mol.spin = 2
    mol.verbose = 0
    mol.output = '/dev/null'
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = xc
    mf.kernel()

    td = build_forge_tddft_object(mf, extype=extype, collinear_samples=collinear_samples, nstates=3)
    nac = packaged_nac.NAC(td).kernel(state_I=state_i, state_J=state_j, use_etfs=use_etfs, ediff=False)
    assert nac.shape == (mol.natm, 3)
    assert np.isfinite(nac).all()
    print(f'Forge XY smoke: XC={xc} use_etfs={use_etfs} shape={nac.shape}')


def run_direct_function_equivalence_case(xc, extype=1, collinear_samples=50, state_i=1, state_j=2):
    mol = gto.Mole()
    mol.atom = '''
O     0.000000    0.000000    0.000000
H     0.000000   -0.757000    0.587000
H     0.000000    0.757000    0.587000
'''
    mol.basis = '631g'
    mol.spin = 2
    mol.verbose = 0
    mol.output = '/dev/null'
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = xc
    mf.kernel()

    td = build_forge_tddft_object(mf, extype=extype, collinear_samples=collinear_samples, nstates=3)
    nac_obj = packaged_nac.NAC(td)
    nac_obj.state_I = state_i
    nac_obj.state_J = state_j
    nac_obj.use_etfs = False
    nac_obj.ediff = False

    x_y_I = td.xy[state_i - 1]
    x_y_J = td.xy[state_j - 1]
    direct = packaged_nac.get_Hellmann_Feymann(nac_obj, x_y_I, x_y_J)
    direct += packaged_nac.nac_csf(nac_obj, x_y_I, x_y_J) * (td.e[state_j - 1] - td.e[state_i - 1])
    via_class = nac_obj.kernel()

    np.testing.assert_allclose(via_class, direct, rtol=1e-9, atol=1e-9)
    print(f'Direct equivalence: XC={xc} max_abs_diff={np.max(np.abs(via_class - direct)):.12e}')


def main():
    for xc in ('HF', 'B3LYP'):
        for use_etfs in (True, False):
            run_case(xc=xc, use_etfs=use_etfs)
    for use_etfs in (True, False):
        run_forge_format_smoke_case(xc='B3LYP', use_etfs=use_etfs)
    for xc in ('HF', 'B3LYP'):
        run_direct_function_equivalence_case(xc=xc)
    print('All tests passed.')


if __name__ == '__main__':
    main()
