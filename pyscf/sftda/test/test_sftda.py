# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
from pyscf import lib, gto
from pyscf import sftda

lib.num_threads(4)

def diagonalize_tda(mf, extype=1, collinear_samples=50, nstates=5):
    a, b = sftda.uhf_sf.get_ab_sf(mf, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b
    n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
    n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]
    A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b, n_occ_a*n_virt_b), order='C')
    B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b, n_occ_b*n_virt_a), order='C')
    B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a, n_occ_a*n_virt_b), order='C')
    A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a, n_occ_b*n_virt_a), order='C')
    Casida_matrix = np.block([[ A_abab_2d, np.zeros_like(B_abba_2d)],
                              [np.zeros_like(-B_baab_2d),-A_baba_2d]])
    eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)
    idx = eigenvals.real.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    norms = np.linalg.norm(eigenvecs[:n_occ_a*n_virt_b], axis=0)**2
    norms -= np.linalg.norm(eigenvecs[n_occ_a*n_virt_b:], axis=0)**2
    if extype == 1:
        mask = norms > 1e-3
        valid_e = eigenvals[mask].real
    else: 
        mask = norms < -1e-3
        valid_e = eigenvals[mask].real
        valid_e = -valid_e
    lowest_e = np.sort(valid_e)[:nstates]
    return lowest_e

def diagonalize_sf_model(mf, extype=1, collinear_samples=50, nstates=5, tda=False):
    a, b = sftda.uhf_sf.get_ab_sf(mf, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b
    n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
    n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]
    A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b, n_occ_a*n_virt_b), order='C')
    B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b, n_occ_b*n_virt_a), order='C')
    B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a, n_occ_a*n_virt_b), order='C')
    A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a, n_occ_b*n_virt_a), order='C')
    if tda:
        B_abba_2d[:] = 0
        B_baab_2d[:] = 0
    h = np.block([[ A_abab_2d, B_abba_2d],
                  [-B_baab_2d,-A_baba_2d]])   
    eigenvals, eigenvecs = np.linalg.eig(h)
    idx = eigenvals.real.argsort()
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    norms = np.linalg.norm(eigenvecs[:n_occ_a * n_virt_b], axis=0)**2
    norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b:], axis=0)**2
    if extype == 1:
        mask = norms > 1e-3
        valid_e = eigenvals[mask].real
    else: 
        mask = norms < -1e-3
        valid_e = eigenvals[mask].real
        valid_e = -valid_e
    lowest_e = np.sort(valid_e)[:nstates]
    return lowest_e


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        # mol.output = '/dev/null'
        mol.atom = '''
        O     0.   0.       0.
        H     0.   -0.757   0.587
        H     0.   0.757    0.587'''
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_hf_tda(self):
        mf = self.mol.UKS(xc='HF').run()
        ref = np.array([0.4664408971, 0.5575560788, 1.0531058659])
        td = mf.TDA_SF().set(extype=0, collinear_samples=50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=0, collinear_samples=50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

        ref = np.array([-0.2157451986, 0.0027042225, 0.0314394519])
        td = mf.TDA_SF().set(extype=1, collinear_samples=50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=1, collinear_samples=50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

    def test_mcol_lda_tda(self):
        mf = self.mol.UKS(xc='SVWN').run()
        ref = np.array([0.4502240188, 0.5791758572, 1.0447533763])
        td = mf.TDA_SF().set(extype=0, collinear_samples=50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=0, collinear_samples=50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

        ref = np.array([-0.3264298143, 0.0003751741, 0.0215670204])
        td = mf.TDA_SF().set(extype=1, collinear_samples=50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=1, collinear_samples=50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

    def test_mcol_b3lyp_tda(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        ref = np.array([0.4594126525, 0.5779958668, 1.0662926327])
        td = mf.TDA_SF().set(extype=0, collinear_samples=50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=0, collinear_samples=50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

        ref = np.array([-0.2962917674, 0.0006700177, 0.0195628987])
        td = mf.TDA_SF().set(extype=1, collinear_samples=50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=1, collinear_samples=50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

    def test_col_b3lyp_tda(self):
        mf = self.mol.UKS(xc='B3LYP').run()
        ref = np.array([0.4737123152, 0.6066070401, 1.0843696957])
        td = mf.TDA_SF().set(extype=0, collinear_samples=-50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=0, collinear_samples=-50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

        ref = np.array([-0.2851971087, 0.0428751344, 0.1026984023])
        td = mf.TDA_SF().set(extype=1, collinear_samples=-50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=1, collinear_samples=-50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

    def test_mcol_tpss_tda(self):
        mf = self.mol.UKS(xc='TPSS').run()
        ref = np.array([0.449865372, 0.5707190159, 1.0544112679])
        td = mf.TDA_SF().set(extype=0, collinear_samples=50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=0, collinear_samples=50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

        ref = np.array([-0.2869994089, 0.0006366278, 0.0232922927])
        td = mf.TDA_SF().set(extype=1, collinear_samples=50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=1, collinear_samples=50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
    
    def test_mcol_cam_tda(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        ref = np.array([0.4617250119, 0.5771124927, 1.0702239617])
        td = mf.TDA_SF().set(extype=0, collinear_samples=50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=0, collinear_samples=50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

        ref = np.array([-0.2975653443, 0.0006832701, 0.019789413])
        td = mf.TDA_SF().set(extype=1, collinear_samples=50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=1, collinear_samples=50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)
    
    def test_col_cam_tda(self):
        mf = self.mol.UKS(xc='CAM-B3LYP').run()
        ref = np.array([0.4754934156, 0.6049670138, 1.0891712805])
        td = mf.TDA_SF().set(extype=0, collinear_samples=-50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=0, collinear_samples=-50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

        ref = np.array([-0.2873952196, 0.0303549298, 0.0990148165])
        td = mf.TDA_SF().set(extype=1, collinear_samples=-50, nstates=3).run()
        self.assertAlmostEqual(abs(td.e - ref).max(), 0, 6)
        e = diagonalize_tda(mf, extype=1, collinear_samples=-50, nstates=3)
        self.assertAlmostEqual(abs(e - td.e).max(), 0, 6)

if __name__ == "__main__":
    print("Full Tests for spin-flip-TDA with multicollinear functionals and collinear functionals")
    unittest.main()