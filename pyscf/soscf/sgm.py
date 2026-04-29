#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

'''
Square Gradient Minimization (SGM) for excited-state SCF orbital optimization.

SGM minimizes the squared norm of the orbital gradient Delta = g^T @ g,
turning the saddle-point search for excited states into a minimization problem.
It uses L-BFGS with a lightweight backtracking line search.

Ref: J. Chem. Theory Comput. 2020, 16, 3, 1699
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import hf_symm, uhf_symm, ghf_symm
from pyscf.soscf.newton_ah import gen_g_hop_rohf
from pyscf import __config__


class LBFGSMemory:
    '''Bounded-memory storage for L-BFGS (s_k, y_k, rho_k) pairs.

    L-BFGS approximates the inverse Hessian from the last m steps.
    s_k = x_{k+1} - x_k (position difference)
    y_k = grad_{k+1} - grad_k (gradient difference)
    rho_k = 1 / (y_k^T @ s_k)
    '''
    def __init__(self, m=8):
        self.m = m
        self.s_list = []
        self.y_list = []
        self.rho_list = []

    def push(self, s, y):
        '''Add a new (s,y) pair, dropping the oldest if over capacity.'''
        rho = 1.0 / (max(numpy.dot(s, y), 1e-100))
        self.s_list.append(s.copy())
        self.y_list.append(y.copy())
        self.rho_list.append(rho)
        if len(self.s_list) > self.m:
            self.s_list.pop(0)
            self.y_list.pop(0)
            self.rho_list.pop(0)

    def clear(self):
        '''Reset memory (e.g., after line search failure).'''
        self.s_list.clear()
        self.y_list.clear()
        self.rho_list.clear()

    def __len__(self):
        return len(self.s_list)


def lbfgs_two_loop(grad, s_list, y_list, rho_list, H0_inv):
    '''L-BFGS two-loop recursion: p = -H_approx^{-1} @ grad.

    The initial inverse Hessian guess is H0_inv (diagonal preconditioner).
    This is applied element-wise at step 3 of the two-loop recursion,
    replacing the standard scaled identity guess (gamma * I) used in
    traditional BFGS. The SGM preconditioner H0_inv = 1 / (8*eps_ai^2 + 1e-12)
    exploits the quadratic dependence of Delta's Hessian on orbital energies.

    Returns:
        p: search direction (descent direction, so p^T @ grad < 0)
    '''
    k = len(s_list)
    alpha = numpy.zeros(k)
    q = grad.copy()

    # First loop (backward over stored pairs)
    for i in range(k - 1, -1, -1):
        alpha[i] = rho_list[i] * numpy.dot(s_list[i], q)
        q -= alpha[i] * y_list[i]

    # Initial inverse Hessian: diagonal preconditioner
    z = H0_inv * q

    # Second loop (forward)
    for i in range(k):
        beta = rho_list[i] * numpy.dot(y_list[i], z)
        z += (alpha[i] - beta) * s_list[i]

    return -z


def get_g_orb_rohf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None, with_symmetry=True):
    '''CHEAP: compute only the ROHF orbital gradient and h_diag.

    Builds the Fock matrix from density and extracts the off-diagonal
    occupied-virtual gradient. This is a single SCF cycle — it does NOT
    call mf.gen_response() and does NOT build the Hessian-vector product
    closure h_op. Used during backtracking line search for fast trial
    step evaluation.

    The computation mirrors gen_g_hop_rohf up to the gradient extraction
    (before mf.gen_response()), then applies the ROHF sum_ab reduction.

    Returns:
        g_orb: 1D array, packed ROHF orbital gradient
    '''
    if getattr(fock_ao, 'focka', None) is None:
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
    fock_ao = fock_ao.focka, fock_ao.fockb
    mo_occa = occidxa = mo_occ > 0
    mo_occb = occidxb = mo_occ ==2
    ug = gen_g_orb_uhf(mf, (mo_coeff,)*2, (mo_occa,mo_occb), 
                       fock_ao, None, with_symmetry)

    viridxa = ~occidxa
    viridxb = ~occidxb
    uniq_var_a = viridxa[:,None] & occidxa
    uniq_var_b = viridxb[:,None] & occidxb
    uniq_ab = uniq_var_a | uniq_var_b
    nmo = mo_coeff.shape[-1]
    nocca = numpy.count_nonzero(mo_occa)
    nvira = nmo - nocca

    def sum_ab(x):
        x1 = numpy.zeros((nmo,nmo), dtype=x.dtype)
        x1[uniq_var_a]  = x[:nvira*nocca]
        x1[uniq_var_b] += x[nvira*nocca:]
        return x1[uniq_ab]
    g_orb = sum_ab(ug)

    return g_orb

def gen_g_orb_uhf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None,
                  with_symmetry=True):
    mol = mf.mol
    mo_coeff0 = mo_coeff

    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]
    if with_symmetry and mol.symmetry:
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        sym_forbida = orbsyma[viridxa,None] != orbsyma[occidxa]
        sym_forbidb = orbsymb[viridxb,None] != orbsymb[occidxb]
        sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

    if fock_ao is None:
        if getattr(mf, '_scf', None) and mf._scf.mol != mol:
            h1e = mf.get_hcore(mol)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
        focka = reduce(numpy.dot, (mo_coeff[0].conj().T, fock_ao[0], mo_coeff[0]))
        fockb = reduce(numpy.dot, (mo_coeff[1].conj().T, fock_ao[1], mo_coeff[1]))
    else:
        focka = reduce(numpy.dot, (mo_coeff0[0].conj().T, fock_ao[0], mo_coeff0[0]))
        fockb = reduce(numpy.dot, (mo_coeff0[1].conj().T, fock_ao[1], mo_coeff0[1]))

    g = numpy.hstack((focka[viridxa[:,None],occidxa].ravel(),
                      fockb[viridxb[:,None],occidxb].ravel()))
    if with_symmetry and mol.symmetry:
        g[sym_forbid] = 0
    return g

class _SGM:
    '''Square Gradient Minimization mixin for ROHF.

    Attributes:
        max_cycle : int
            Maximum SGM macro-iterations. Default 100.
        tol : float
            Convergence threshold on sqrt(Delta). Default 1e-6.
        lbfgs_memory : int
            Number of (s,y) pairs stored. Default 8.
        c1 : float
            Armijo condition constant. Default 1e-4.
        min_step : float
            Minimum step size in line search. Default 1e-4.
        first_step_scale : float
            Scale factor for the first L-BFGS step. Default 0.1.
    '''

    max_cycle = getattr(__config__, 'sgm_max_cycle', 100)
    tol = getattr(__config__, 'sgm_tol', 1e-4)
    lbfgs_memory = getattr(__config__, 'sgm_lbfgs_memory', 8)
    c1 = getattr(__config__, 'sgm_c1', 1e-4)
    min_step = getattr(__config__, 'sgm_min_step', 1e-4)
    first_step_scale = getattr(__config__, 'sgm_first_step_scale', 1)
    canonicalization = getattr(__config__, 'soscf_newton_ah_SOSCF_canonicalization', True)

    _keys = {
        'max_cycle', 'tol', 'lbfgs_memory', 'c1', 'min_step', 'first_step_scale', 'canonicalization'
    }

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self._scf = mf

    # ---- Cheap gradient: Fock build only, no response kernel ----
    get_grad = staticmethod(get_g_orb_rohf)

    # ---- Expensive: full gen_g_hop from newton_ah (includes response kernel) ----
    gen_g_hop = staticmethod(gen_g_hop_rohf)

    def kernel(self, mo_coeff=None, mo_occ=None):
        '''SGM optimization kernel.

        Args:
            mo_coeff: initial MO coefficients (nao, nmo).
                      If None, use self.mo_coeff.
            mo_occ: occupation numbers (nmo,). Must be provided for
                    excited state targeting.

        Returns:
            conv: bool, convergence flag
            Delta: float, final squared gradient norm
            mo_coeff: optimized MO coefficients
            mo_occ: unchanged occupation numbers
        '''
        log = logger.new_logger(self, self.verbose)
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        if mo_occ is None:
            raise RuntimeError('mo_occ must be specified for SGM')

        h1e = self._scf.get_hcore()
        dm = self._scf.make_rdm1(mo_coeff, mo_occ)
        vhf = self._scf.get_veff(self._scf.mol, dm)
        e_tot = self._scf.energy_tot(dm, h1e, vhf)

        t0 = (logger.process_clock(), logger.perf_counter())

        # ---- Initialization: full state (EXPENSIVE) ----
        g_orb, h_op, h_diag = self.gen_g_hop(self._scf, mo_coeff, mo_occ)
        Delta = numpy.dot(g_orb, g_orb)
        grad_Delta = 2.0 * h_op(g_orb)

        mem = LBFGSMemory(m=self.lbfgs_memory)

        log.info('SGM: initial Delta = %.6g, |g| = %.6g', Delta, numpy.sqrt(Delta))

        conv = False
        for it in range(self.max_cycle):
            norm_g = numpy.sqrt(Delta)
            if norm_g < self.tol:
                log.info('SGM converged in %d iterations: |g| = %.6g < %.6g',
                         it, norm_g, self.tol)
                conv = True
                break

            # ---- Step 1: Build preconditioner ----
            # SGM-preferred: 1 / (8 * epsilon_ai^2 + 1e-12)
            # H0_inv = 1.0 / (8.0 * h_diag**2 + 1e-12)
            # different from the original paper, but seems to work better in practice
            H0_inv = 1.0 / (2.0 * h_diag**2 + 1e-12)

            # ---- Step 2: L-BFGS search direction ----
            p = lbfgs_two_loop(grad_Delta,
                               mem.s_list, mem.y_list, mem.rho_list,
                               H0_inv)

            # Scale first step to avoid wild initial guess
            if it == 0:
                p *= self.first_step_scale

            # ---- Step 3: Backtracking line search (only CHEAP evaluations) ----
            alpha = 1.0
            direction_dot_grad = numpy.dot(p, grad_Delta)

            # Fallback: if L-BFGS gives a non-descent direction, reset memory
            # and use preconditioned steepest descent
            if direction_dot_grad > 0:
                log.warn('SGM: non-descent direction at iter %d, resetting L-BFGS', it)
                mem.clear()
                p = -grad_Delta * H0_inv
                direction_dot_grad = numpy.dot(p, grad_Delta)

            trial_state = None
            step = None

            while alpha > self.min_step:
                step = alpha * p

                # Convert step vector to anti-symmetric kappa, then unitary U
                kappa = hf.unpack_uniq_var(step, mo_occ)
                U = scipy.linalg.expm(kappa)
                C_trial = numpy.dot(mo_coeff, U)
                if self._scf.mol.symmetry:
                    orbsym = hf_symm.get_orbsym(self._scf.mol, mo_coeff)
                    C_trial = lib.tag_array(C_trial, orbsym=orbsym)

                # CHEAP: only compute Delta (1 Fock build, no response kernel)
                g_trial = self.get_grad(self._scf, C_trial, mo_occ)
                Delta_trial = numpy.dot(g_trial, g_trial)

                # Armijo sufficient descent condition
                if Delta_trial <= Delta + self.c1 * alpha * direction_dot_grad:
                    log.debug2('SGM: step accepted, alpha=%.4g, Delta_trial=%.6g',
                               alpha, Delta_trial)

                    dm_new = self._scf.make_rdm1(C_trial, mo_occ)
                    vhf_new = self._scf.get_veff(self._scf.mol, dm_new)
                    e_tot_new = self._scf.energy_tot(dm_new, h1e, vhf_new)
                    delta_E = e_tot_new - e_tot
                    e_tot = e_tot_new  # 更新当前能量
                    vhf = vhf_new      # 保存 vhf 供收尾使用

                    # Step accepted — do the ONE expensive gradient evaluation
                    g_new, h_op_new, h_diag_new = self.gen_g_hop(
                        self._scf, C_trial, mo_occ)
                    grad_Delta_new = 2.0 * h_op_new(g_new)

                    trial_state = (C_trial, g_new, h_op_new, h_diag_new,
                                   Delta_trial, grad_Delta_new)
                    break
                else:
                    alpha *= 0.5

            # ---- Step 4: Fail-safe ----
            if trial_state is None:
                log.warn('SGM: line search failed at iteration %d, '
                         'resetting L-BFGS history', it)
                mem.clear()
                continue

            # ---- Step 5: Update L-BFGS memory ----
            C_trial, g_new, h_op_new, h_diag_new, Delta_trial, grad_Delta_new = \
                trial_state
            s_k = step
            y_k = grad_Delta_new - grad_Delta

            if numpy.dot(s_k, y_k) > 1e-10:
                mem.push(s_k, y_k)
            else:
                log.debug1('SGM: skipping (s,y) pair — curvature too small')

            # ---- Step 6: Advance state ----
            mo_coeff = C_trial
            g_orb = g_new
            h_op = h_op_new
            h_diag = h_diag_new
            Delta = Delta_trial
            grad_Delta = grad_Delta_new

            log.info('SGM iter %3d: E = %.15g  dE = %g  Delta = %.3g  |g| = %.3g  alpha = %.4g',
                     it, e_tot, delta_E, Delta, numpy.sqrt(Delta), alpha)

        log.timer('SGM optimization', *t0)

        if not conv:
            log.info('SGM did not converge in %d iterations: |g| = %.6g',
                     self.max_cycle, numpy.sqrt(Delta))

        # Store results on self
        dm = self._scf.make_rdm1(mo_coeff, mo_occ)
        fock = self._scf.get_fock(h1e, self._scf.get_ovlp(), vhf, dm)
        mo_energy, mo_coeff_canon = self._scf.canonicalize(mo_coeff, mo_occ, fock)
        if self.canonicalization:
            log.info('Canonicalize SCF orbitals')
            mo_coeff = mo_coeff_canon
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.converged = conv
        self.mo_energy = mo_energy
        self.e_tot = self._scf.energy_tot(dm, h1e, vhf)
        self._finalize()
        return self.e_tot

    def undo_sgm(self):
            '''Remove the SGM Mixin and return the original SCF object.'''
            obj = lib.view(self, self._scf.__class__)
            del obj._scf
            return obj


def SGM(mf):
    '''Create SGM optimizer for the given SCF object.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.2', basis='sto-3g')
    >>> mf = scf.ROHF(mol).run()
    >>> mo_occ = mf.mo_occ.copy()
    >>> homo = (mo_occ > 0).sum() - 1
    >>> mo_occ[homo] = 1  # singly occupied
    >>> mo_occ[homo+1] = 1  # singly occupied
    >>> sgm_mf = SGM(mf)
    >>> sgm_mf.kernel(mf.mo_coeff, mo_occ)
    '''
    if isinstance(mf, _SGM):
        return mf

    assert isinstance(mf, hf.SCF)
    return lib.set_class(_SGM(mf), (_SGM, mf.__class__))


# Hook into pyscf.lib so SGM can access set_class
from pyscf import lib  # noqa

if __name__ == '__main__':
    '''
    A difficult example to show the overperformance of SGM over MOM.
    '''
    from pyscf import gto, scf
    atom = '''
    C 0.00000000 0.00000000 -1.13947666
    O 0.00000000 0.00000000 1.14402883
    H 0.00000000 1.76627623 -2.23398653
    H 0.00000000 -1.76627623 -2.23398653
    '''
    mol = gto.M(atom=atom, unit='B', charge=0, spin=0, basis='aug-cc-pvtz', symmetry=True)
    mf0 = mol.ROKS(xc='HF')
    mf0.kernel()
    mf0.analyze(verbose=6)

    setocc = mf0.to_uks().mo_occ
    setocc[1][7] -= 1  # hole in n orbital
    setocc[0][14] += 1  # electron in C4pz* orbital
    print(setocc)
    ro_occ = setocc[0] + setocc[1]
    print(ro_occ)

    mol.spin = 2
    mf3 = mol.ROKS(xc='HF')
    mf3.verbose = 4
    dm0 = mf3.make_rdm1(mf0.mo_coeff, ro_occ)
    mf3 = scf.addons.mom_occ(mf3, mf0.mo_coeff, setocc)
    mf3.kernel(dm0)

    from pyscf.soscf import sgm
    mf4 = mol.ROKS(xc='HF')
    mf4.verbose = 4
    mf4.mo_coeff = mf0.mo_coeff
    mf4.mo_occ = ro_occ
    sgm_mf = sgm.SGM(mf4)
    sgm_mf.kernel()
