import numpy as np
from sympy import S
from sympy.physics.quantum.cg import CG
from pyscf import gto, lib, sftda
from pyscf.lib import logger
from pyscf.scf.jk import get_jk
from pyscf.data.nist import HARTREE2WAVENUMBER, LIGHT_SPEED, HARTREE2EV

def cg(j1, m1, j2, m2, j, m):
    """
    Calculate Clebsch-Gordan coefficient using SymPy
    """
    j1, m1 = S(j1), S(m1)
    j2, m2 = S(j2), S(m2)
    j, m = S(j), S(m)

    cg_coeff = CG(j1, m1, j2, m2, j, m)

    result = float(cg_coeff.doit())
    return result

def sozeff(atom, zeff_type="orca"):
    """
    Calculate effective nuclear charge for given atomic number
    copyed from: https://github.com/masaya0222/PyGraSO/blob/main/pygraso/calc_ao_element.py
    """
    assert zeff_type in ["one", "orca", "pysoc"], f"{zeff_type} is not valid"
    neval = {
        1: 1,
        2: 2,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
        8: 6,
        9: 7,
        10: 8,
        11: 1,
        12: 2,
        13: 3,
        14: 4,
        15: 5,
        16: 6,
        17: 7,
        18: 8,
        19: 1,
        20: 2,
        21: 3,
        22: 4,
        23: 5,
        24: 6,
        25: 7,
        26: 8,
        27: 9,
        28: 10,
        29: 11,
        30: 12,
        31: 3,
        32: 4,
        33: 5,
        34: 6,
        35: 7,
        36: 8,
        37: 1,
        38: 2,
        39: 3,
        40: 4,
        41: 5,
        42: 6,
        43: 7,
        44: 8,
        45: 9,
        46: 10,
        47: 11,
        48: 12,
        49: 3,
        50: 4,
        51: 5,
        52: 6,
        53: 7,
        54: 8,
    }
    if zeff_type == "one":
        return atom

    if zeff_type == "pysoc":
        if atom == 1:
            return 1.0
        elif atom == 2:
            return 2.0
        elif 3 <= atom <= 10:
            return (0.2517 + 0.0626 * neval[atom]) * atom
        elif 11 <= atom <= 18:
            return (0.7213 + 0.0144 * neval[atom]) * atom
        elif (19 <= atom <= 20) or (31 <= atom <= 36):
            return (0.8791 + 0.0039 * neval[atom]) * atom
        elif (37 <= atom <= 38) or (49 <= atom <= 54):
            return (0.9228 + 0.0017 * neval[atom]) * atom
        elif atom == 26:
            return 0.583289 * atom
        elif atom == 30:
            return 330.0
        elif 21 <= atom <= 30:
            return atom * (0.385 + 0.025 * (neval[atom] - 2))
        elif 39 <= atom <= 48:
            return atom * (4.680 + 0.060 * (neval[atom] - 2))
        elif atom == 72:
            return 1025.28
        elif atom == 73:
            return 1049.74
        elif atom == 74:
            return 1074.48
        elif atom == 75:
            return 1099.5
        elif atom == 76:
            return 1124.8
        elif atom == 77:
            return 1150.38
        elif atom == 78:
            return 1176.24
        elif atom == 79:
            return 1202.38
        elif atom == 80:
            return 1228.8
        else:
            raise ValueError(f"SOZEFF is not available for atomic number {atom}")
    if zeff_type == "orca":
        if atom == 1:
            return 1.0
        elif atom == 2:
            return 2.0
        elif 3 <= atom < 10:
            return (0.4 + 0.05 * neval[atom]) * atom
        elif 11 <= atom <= 18:
            return (0.925 - 0.0125 * neval[atom]) * atom
        elif 32 <= atom <= 35:  # Verified from orca output file
            if atom == 32:
                return 32.32
            elif atom == 33:
                return 31.68
            elif atom == 34:
                return 30.94
            elif atom == 35:
                return 30.10
        else:
            raise ValueError(f"SOZEFF is not available for atomic number {atom}")

def calc_ao_soc_1e(mol, Z='one'):
    '''
    The one-body part of Hsoc operator with (effective) nuclear charge.
    '''
    zeff_list = [sozeff(mol.atom_charge(i), zeff_type=Z) for i in range(mol.natm)]
    ao_soc = np.zeros((3, mol.nao_nr(), mol.nao_nr()), dtype=np.complex128)
    for k in range(mol.natm):
        mol.set_rinv_orig(mol.atom_coord(k))
        ao_soc += (-1.0j) * zeff_list[k] * mol.intor('int1e_prinvxp')
    ao_soc /= (2.0 * LIGHT_SPEED**2)
    ao_soc_1 = -0.5 * (ao_soc[0] + 1j * ao_soc[1])
    ao_soc_0 = np.sqrt(0.5) * ao_soc[2]
    ao_soc_m1 = 0.5 * (ao_soc[0] - 1j * ao_soc[1])
    return np.array([ao_soc_m1, ao_soc_0, ao_soc_1])

def calc_ao_soc_1e_x2camf(mol):
    try:
        from socutils.somf import somf_pt
    except ImportError:
        raise ImportError("Please install socutils package to use X2CAMF SOC integrals.")
    ao_soc = 2j * somf_pt.get_psoc_x2camf(mol)
    ao_soc_1 = -0.5 * (ao_soc[0] + 1j * ao_soc[1])
    ao_soc_0 = np.sqrt(0.5) * ao_soc[2]
    ao_soc_m1 = 0.5 * (ao_soc[0] - 1j * ao_soc[1])
    return np.array([ao_soc_m1, ao_soc_0, ao_soc_1])

def calc_ao_soc_1e_x2camf_xresp(mol):
    try:
        from socutils.somf import somf_pt
    except ImportError:
        raise ImportError("Please install socutils package to use X2CAMF SOC integrals.")
    ao_soc = 2j * somf_pt.get_psoc_x2camf(mol, xresp=True)
    ao_soc_1 = -0.5 * (ao_soc[0] + 1j * ao_soc[1])
    ao_soc_0 = np.sqrt(0.5) * ao_soc[2]
    ao_soc_m1 = 0.5 * (ao_soc[0] - 1j * ao_soc[1])
    return np.array([ao_soc_m1, ao_soc_0, ao_soc_1])

def calc_ao_soc_2e(mf):
    '''
    The two-electron part of the Hsoc under the SOMF approximation.
    mf: an instance of pyscf.scf.UHF or pyscf.scf.RHF
    '''
    dm = mf.make_rdm1()

    sso = 1j * mf.mol.intor('int2e_p1vxp1') / (2.0 * LIGHT_SPEED**2)
    sso_1 = -0.5 * (sso[0] + 1j * sso[1])
    sso_0 = np.sqrt(0.5) * sso[2]
    sso_m1 = 0.5 * (sso[0] - 1j * sso[1])

    if dm.ndim==2:
        dmaa = dmbb = 0.5 * dm
    else:
        dmaa, dmbb = dm
    soc_somf_1 = np.einsum('rs,uvrs->uv', dmaa, sso_1) - \
                 np.einsum('rs,ursv->uv', dmaa, sso_1) - \
                 2 * np.einsum('rs,svur->uv', dmaa, sso_1) + \
                 np.einsum('rs,uvrs->uv', dmbb, sso_1) - \
                 2 * np.einsum('rs,ursv->uv', dmbb, sso_1) - \
                 np.einsum('rs,svur->uv', dmbb, sso_1)
    soc_somf_0 = np.einsum('rs,uvrs->uv', dmaa+dmbb, sso_0) - \
                 1.5 * np.einsum('rs,ursv->uv', dmaa+dmbb, sso_0) - \
                 1.5 * np.einsum('rs,svur->uv', dmaa+dmbb, sso_0)
    soc_somf_m1 = np.einsum('rs,uvrs->uv', dmaa, sso_m1) - \
                  2 * np.einsum('rs,ursv->uv', dmaa, sso_m1) - \
                  np.einsum('rs,svur->uv', dmaa, sso_m1) + \
                  np.einsum('rs,uvrs->uv', dmbb, sso_m1) - \
                  np.einsum('rs,ursv->uv', dmbb, sso_m1) - \
                  2 * np.einsum('rs,svur->uv', dmbb, sso_m1)
    soc_somf_1, soc_somf_m1 = 0.5 * (soc_somf_1 + soc_somf_m1.conj()), 0.5 * (soc_somf_m1 + soc_somf_1.conj())
    return np.array([soc_somf_m1, soc_somf_0, soc_somf_1])

def calc_ao_soc_2e_direct(mf):
    '''
    The two-electron part of the Hsoc under the SOMF approximation using direct SCF (get_jk).
    Memory efficient version.
    '''
    mol = mf.mol
    dm = mf.make_rdm1()
    
    # 处理密度矩阵
    if dm.ndim == 2:
        dmaa = dmbb = 0.5 * dm
    else:
        dmaa, dmbb = dm

    # 1. 准备 get_jk 需要的参数
    # 我们需要对 dmaa 和 dmbb 分别计算三种收缩
    # J_term:  'ijkl,lk->ij' (对应 uvrs)
    # K1_term: 'ijkl,jk->il' (对应 ursv)
    # K2_term: 'ijkl,li->kj' (对应 svur)
    
    dm_list = [dmaa, dmaa, dmaa, dmbb, dmbb, dmbb]
    scripts = [
        'ijkl,lk->ij', # J for dmaa
        'ijkl,jk->il', # K1 for dmaa
        'ijkl,li->kj', # K2 for dmaa
        'ijkl,lk->ij', # J for dmbb
        'ijkl,jk->il', # K1 for dmbb
        'ijkl,li->kj'  # K2 for dmbb
    ]
    
    # 2. 调用 get_jk
    # int2e_p1vxp1 有 3 个分量 (x, y, z)，comp=3
    # aosym='s1' 非常重要，因为 SO 算符破坏了置换对称性
    v_matrices = get_jk(mol, dm_list, scripts=scripts,
                        intor='int2e_p1vxp1', comp=3, aosym='a2ij')
    
    # v_matrices 是一个列表，每个元素形状为 (3, nao, nao)，对应 (x, y, z)
    vj_aa, vk1_aa, vk2_aa = v_matrices[0:3]
    vj_bb, vk1_bb, vk2_bb = v_matrices[3:6]

    # 3. 按照原公式组装 (线性组合)
    # 注意：原代码是在计算 sso (复数球张量) 之后再 einsum。
    # 根据线性性质，我们可以先组合 J/K 矩阵，最后再转换成球张量。
    
    # --- 组合 Cartesian 分量的中间体 ---
    
    # soc_somf_1 的系数逻辑:
    # Term 1 (dmaa): +1 * J - 1 * K1 - 2 * K2
    # Term 2 (dmbb): +1 * J - 2 * K1 - 1 * K2
    v_cart_1 = (vj_aa - vk1_aa - 2 * vk2_aa) + \
               (vj_bb - 2 * vk1_bb - vk2_bb)

    # soc_somf_0 的系数逻辑 (dmaa + dmbb):
    # +1 * J - 1.5 * K1 - 1.5 * K2
    v_cart_0 = (vj_aa + vj_bb) - 1.5 * (vk1_aa + vk1_bb) - 1.5 * (vk2_aa + vk2_bb)

    # soc_somf_m1 的系数逻辑:
    # Term 1 (dmaa): +1 * J - 2 * K1 - 1 * K2
    # Term 2 (dmbb): +1 * J - 1 * K1 - 2 * K2
    v_cart_m1 = (vj_aa - 2 * vk1_aa - vk2_aa) + \
                (vj_bb - vk1_bb - 2 * vk2_bb)

    # 4. 将 Cartesian (x, y, z) 转换为 Spherical (+1, 0, -1)
    # 原代码中的 sso 定义:
    # sso_raw = 1j * intor / (2 * c^2)
    # sso_1  = -0.5 * (sso_x + 1j * sso_y)
    # sso_0  = sqrt(0.5) * sso_z
    # sso_m1 = 0.5 * (sso_x - 1j * sso_y)
    
    # 提取分量 (注意：get_jk 算出来的只是 intor 部分，没有 prefactor)
    prefactor = 1j / (2.0 * LIGHT_SPEED**2)
    
    def to_spherical(v_cart_xyz, component):
        # v_cart_xyz shape is (3, nao, nao)
        vx = v_cart_xyz[0]
        vy = v_cart_xyz[1]
        vz = v_cart_xyz[2]
        
        if component == 1:
            return -0.5 * (vx + 1j * vy)
        elif component == 0:
            return np.sqrt(0.5) * vz
        elif component == -1:
            return 0.5 * (vx - 1j * vy)

    soc_somf_1 = to_spherical(v_cart_1, 1) * prefactor
    soc_somf_0 = to_spherical(v_cart_0, 0) * prefactor
    soc_somf_m1 = to_spherical(v_cart_m1, -1) * prefactor

    # 5. 厄米化处理 (原代码最后一步)
    soc_somf_1, soc_somf_m1 = 0.5 * (soc_somf_1 + soc_somf_m1.conj()), \
                              0.5 * (soc_somf_m1 + soc_somf_1.conj())
    
    # 注意：soc_somf_0 理论上也应该反对称或厄米，原代码没写，这里保持原样
    # 如果需要保证 output 结构一致，可以加上 return
    return np.array([soc_somf_m1, soc_somf_0, soc_somf_1])

def calc_soc(soc_ao, mf, xy1, xy2, s1, s2, log=None):
    sz = 0.5 * mf.mol.spin - 1

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = mo_occ[0] == 0
    viridxb = mo_occ[1] == 0
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:, viridxa]
    orbvb = mo_coeff[1][:, viridxb]
    mx = xy1[0]
    nx = xy2[0]
    if isinstance(xy1[1], np.ndarray):
        my = xy1[1]
        ny = xy2[1]
    else:
        ny = None
        my = None
    gamma_oo_aa = - lib.einsum('ia,ja->ij', mx.conj(), nx)
    gamma_aa = lib.einsum('uj,vi,ij->vu', orboa.conj(), orboa, gamma_oo_aa)
    gamma_vv_bb = lib.einsum('ib,ia->ab', mx.conj(), nx)
    gamma_bb = lib.einsum('ub,va,ab->vu', orbvb.conj(), orbvb, gamma_vv_bb)
    if my is not None:
        gamma_oo_bb = - lib.einsum('ja,ia->ij', my.conj(), ny)
        gamma_bb += lib.einsum('uj,vi,ij->vu', orbob.conj(), orbob, gamma_oo_bb)
        gamma_vv_aa = lib.einsum('ia,ib->ab', my.conj(), ny)
        gamma_aa += lib.einsum('ub,va,ab->vu', orbva.conj(), orbva, gamma_vv_aa)

    cg_coeff = cg(s2, sz, 1, 0, s1, sz)
    if log is not None:
        log.info('Clebsch-Gordan coefficient: <%3.1f,%3.1f;%3.1f,%3.1f|%3.1f,%3.1f> = %.3f',
                  s2, sz, 1., 0., s1, sz, cg_coeff)
    if abs(cg_coeff)<1e-8:
        return np.zeros((int(2*s1)+1, int(2*s2)+1), dtype=np.complex128)
    gamma = (gamma_aa - gamma_bb) / np.sqrt(2) / cg_coeff
    result = np.zeros((int(2*s1)+1, int(2*s2)+1), dtype=np.complex128)
    for m1 in np.arange(-s1, s1+1):
        for m2 in np.arange(-s2, s2+1):
            if m1-m2==1:
                tmp = np.einsum('vu,uv->', gamma, soc_ao[0])
                tmp *= - cg(s2, m2, 1, 1, s1, m1)
                result[int(m1+s1), int(m2+s2)] = tmp
            elif m1-m2==0:
                tmp = np.einsum('vu,uv->', gamma, soc_ao[1])
                tmp *= cg(s2, m2, 1, 0, s1, m1)
                result[int(m1+s1), int(m2+s2)] = tmp
            elif m1-m2==-1:
                tmp = np.einsum('vu,uv->', gamma, soc_ao[2])
                tmp *= - cg(s2, m2, 1, -1, s1, m1)
                result[int(m1+s1), int(m2+s2)] = tmp
    return result

def _get_soc_ao(mf, soctype):
    mol = mf.mol
    if soctype == 'SOMF':
        soc_ao = calc_ao_soc_1e(mol, Z='one')
        soc_ao += calc_ao_soc_2e_direct(mf)
    elif soctype == 'Zeff':
        soc_ao = calc_ao_soc_1e(mol, Z='orca')
    elif soctype == '1e':
        soc_ao = calc_ao_soc_1e(mol, Z='one')
    elif soctype == 'X2CAMF':
        soc_ao = calc_ao_soc_1e_x2camf(mol)
    elif soctype == 'X2CAMF_XRESP':
        soc_ao = calc_ao_soc_1e_x2camf_xresp(mol)
    else:
        raise ValueError(f'soctype={soctype} is not supported.')
    return soc_ao

def _spin_from_s2(s2):
    return round(-1 + np.sqrt(1 + 4 * s2)) / 2


def _format_pretty_soc_lines(matrix, s1, s2):
    """
    Format SOC matrix elements with Sz labels.

    Args:
        matrix: np.array, dtype=complex128, shape should be (2*s1+1, 2*s2+1)
        s1: float, total spin S for bra states (rows)
        s2: float, total spin S for ket states (columns)
    """
    n_rows = int(2 * s1 + 1)
    n_cols = int(2 * s2 + 1)

    if matrix.shape != (n_rows, n_cols):
        return [f'Warning: Matrix shape {matrix.shape} mismatch with spins '
                f'S1={s1}, S2={s2} (expected {n_rows}x{n_cols})']

    row_sz_vals = np.linspace(-s1, s1, n_rows)
    col_sz_vals = np.linspace(-s2, s2, n_cols)

    col_width = 25
    label_width = 16

    header = ' ' * label_width
    for sz in col_sz_vals:
        header += f'|Sz={sz:5.2f}>'.center(col_width)

    lines = ['Actual matrix elements (cm^-1):', header, '-' * len(header)]

    for i, row_sz in enumerate(row_sz_vals):
        row_str = f'<Sz={row_sz:5.2f}|'.ljust(label_width)

        for j in range(n_cols):
            val = matrix[i, j]
            val_str = f'({val.real:9.6f},{val.imag:9.6f})'
            row_str += val_str.center(col_width)

        lines.append(row_str)
    return lines


def _log_pretty_soc(log, matrix, s1, s2):
    for line in _format_pretty_soc_lines(matrix, s1, s2):
        log.info(line)


def print_pretty_soc(matrix, s1, s2, verbose=logger.INFO):
    """Format and log SOC matrix elements with Sz labels."""
    log = logger.Logger(verbose=verbose)
    for line in _format_pretty_soc_lines(matrix, s1, s2):
        log.info(line)


def analyze_soc(td, soctype='SOMF', verbose=None):
    '''
    td: an instance of pyscf.sftda.TDA_SF
    '''
    log = logger.new_logger(td, verbose)
    cpu0 = (logger.process_clock(), logger.perf_counter())

    soc_ao = _get_soc_ao(td._scf, soctype)
    cpu0 = log.timer(f'{soctype} SOC AO integrals', *cpu0)

    ss = td.spin_square()
    for ket in range(td.nstates):
        for bra in range(ket+1, td.nstates):
            log.note('SOC matrix elements between states %d and %d', bra+1, ket+1)
            s21 = ss[bra]
            s22 = ss[ket]
            s1 = _spin_from_s2(s21)
            s2 = _spin_from_s2(s22)
            log.note('Bra state %d: S=%s  <S^2>=%.4f; Ket state %d: S=%s  <S^2>=%.4f',
                     bra+1, s1, s21, ket+1, s2, s22)
            soc_mat = calc_soc(soc_ao, td._scf, td.xy[bra], td.xy[ket], s1, s2, log=log)
            _log_pretty_soc(log, soc_mat * HARTREE2WAVENUMBER, s1, s2)
            log.note('SOCC(states %d, %d) = %.6f cm^-1',
                     bra+1, ket+1, np.linalg.norm(soc_mat) * HARTREE2WAVENUMBER)
            log.note('')

    log.timer('SOC analysis', *cpu0)


def build_and_diagonalize_soc(td, soctype='SOMF', verbose=None):
    '''
    Build the full SOC Hamiltonian and diagonalize it.
    td: pyscf.sftda.TDA_SF instance
    '''
    log = logger.new_logger(td, verbose)
    cpu0 = (logger.process_clock(), logger.perf_counter())

    soc_ao = _get_soc_ao(td._scf, soctype)
    cpu0 = log.timer(f'{soctype} SOC AO integrals', *cpu0)

    nstates = td.nstates

    state_info = []  # spin, block offsets, and scalar-state energies
    current_idx = 0

    log.info('Precomputing spin subspaces and block offsets...')
    ss = td.spin_square()
    for i in range(nstates):
        s2_val = ss[i]
        s_val = _spin_from_s2(s2_val)

        dim = int(2 * s_val + 1)

        energy = td.e_tot[i]

        state_info.append({
            'state_id': i,
            's': s_val,
            'dim': dim,
            'start': current_idx,
            'end': current_idx + dim,
            'energy': energy
        })
        current_idx += dim

    total_dim = current_idx
    log.info('Total Hamiltonian dimension: %d x %d', total_dim, total_dim)

    H_soc = np.zeros((total_dim, total_dim), dtype=np.complex128)

    for info in state_info:
        # Only scalar-state energies are included on the diagonal.
        idx_slice = slice(info['start'], info['end'])
        H_soc[idx_slice, idx_slice] = np.eye(info['dim']) * info['energy']

    for i in range(nstates):
        for j in range(i + 1, nstates):
            bra_info = state_info[j]
            ket_info = state_info[i]

            sub_mat = calc_soc(soc_ao, td._scf,
                               td.xy[j], td.xy[i],
                               bra_info['s'], ket_info['s'], log=log)
            log.info('SOC matrix elements between states %d and %d', j+1, i+1)
            _log_pretty_soc(log, sub_mat * HARTREE2WAVENUMBER, bra_info['s'], ket_info['s'])
            log.note('SOCC(states %d, %d) = %.6f cm^-1',
                     j+1, i+1, np.linalg.norm(sub_mat) * HARTREE2WAVENUMBER)

            H_soc[bra_info['start']:bra_info['end'], ket_info['start']:ket_info['end']] = sub_mat

            H_soc[ket_info['start']:ket_info['end'], bra_info['start']:bra_info['end']] = sub_mat.conj().T

    cpu0 = log.timer('SOC Hamiltonian construction', *cpu0)

    log.info('Diagonalizing the SOC Hamiltonian...')
    eigvals, eigvecs = np.linalg.eigh(H_soc)
    cpu0 = log.timer('SOC Hamiltonian diagonalization', *cpu0)

    log.note('')
    log.note('%s', '='*60)
    log.note('Spin-orbit-coupled eigenstates')
    log.note('%s', '='*60)

    min_e = np.min(eigvals)

    for idx, e in enumerate(eigvals):
        rel_e_cm = (e - min_e) * HARTREE2WAVENUMBER
        rel_e_ev = (e - min_e) * HARTREE2EV

        log.note('State %d: Delta E = %.2f cm^-1  (%.6f eV)', idx+1, rel_e_cm, rel_e_ev)

        vec = eigvecs[:, idx]
        norm_sq = np.abs(vec)**2

        log.info('  Dominant scalar-state contributions:')
        for info in state_info:
            prob = np.sum(norm_sq[info['start']:info['end']])
            if prob > 0.01:
                log.info('    %5.1f%% from scalar state %d (S=%s)', prob*100, info['state_id']+1, info['s'])
        log.info('%s', '-' * 30)

    return eigvals, eigvecs, H_soc
