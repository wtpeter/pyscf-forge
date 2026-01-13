import numpy as np
from sympy import S
from sympy.physics.quantum.cg import CG
from pyscf import gto, sftda
from pyscf.scf.jk import get_jk
from pyscf.data.nist import HARTREE2WAVENUMBER, LIGHT_SPEED, HARTREE2EV
from pyscf.sftda.tools_td import spin_square

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
    assert zeff_type in ["one", "orca", "pysoc"], f"{zeff_type=} is not valid"
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

def calc_soc(soc_ao, mf, xy1, xy2, s1, s2):
    '''
    Calculate SOC matrix element between two states
    xy1, xy2: excitation amplitudes of two states
    s1, s2: S values of two states
    '''
    sz = 0.5 * mf.mol.spin - 1
    x1 = xy1[0][1].copy()
    x2 = xy2[0][1].copy()
    soc_mo_aa = mf.mo_coeff[0].T @ soc_ao @ mf.mo_coeff[0]
    soc_mo_bb = mf.mo_coeff[1].T @ soc_ao @ mf.mo_coeff[1]
    soc_mo_aa_oo = soc_mo_aa[:, mf.mo_occ[0]>0][:, :, mf.mo_occ[0]>0]
    soc_mo_bb_vv = soc_mo_bb[:, mf.mo_occ[1]==0][:, :, mf.mo_occ[1]==0]

    print(f'Clebsch-Gordan coefficient: <{s2:.1f},{sz:.1f};{1:.1f},{0:.1f}|{s1:.1f},{sz:.1f}>={cg(s2, sz, 1, 0, s1, sz):.3f}')
    rem0 = -np.einsum('ia,ja->ij', x1, x2) / cg(s2, sz, 1, 0, s1, sz) / np.sqrt(2)
    rem1 = -np.einsum('ia,ib->ab', x1, x2) / cg(s2, sz, 1, 0, s1, sz) / np.sqrt(2)
    result = np.zeros((int(2*s1)+1, int(2*s2)+1), dtype=np.complex128)
    for m1 in np.arange(-s1, s1+1):
        for m2 in np.arange(-s2, s2+1):
            if m1-m2==1:
                tmp = np.einsum('ij,ji->', rem0, soc_mo_aa_oo[0])
                tmp += np.einsum('ab,ab->', rem1, soc_mo_bb_vv[0])
                tmp *= - cg(s2, m2, 1, 1, s1, m1)
                result[int(m1+s1), int(m2+s2)] = tmp
            elif m1-m2==0:
                tmp = np.einsum('ij,ji->', rem0, soc_mo_aa_oo[1])
                tmp += np.einsum('ab,ab->', rem1, soc_mo_bb_vv[1])
                tmp *= cg(s2, m2, 1, 0, s1, m1)
                result[int(m1+s1), int(m2+s2)] = tmp
            elif m1-m2==-1:
                tmp = np.einsum('ij,ji->', rem0, soc_mo_aa_oo[2])
                tmp += np.einsum('ab,ab->', rem1, soc_mo_bb_vv[2])
                tmp *= - cg(s2, m2, 1, -1, s1, m1)
                result[int(m1+s1), int(m2+s2)] = tmp
    return result

def print_pretty_soc(matrix, s1, s2):
    """
    格式化打印 SOC 矩阵，自动生成 Sz 标签并对齐。
    
    Args:
        matrix: np.array, dtype=complex128, 形状应为 (2*s1+1, 2*s2+1)
        s1: float, 左矢 (Bra) 的总自旋 S (用于行)
        s2: float, 右矢 (Ket) 的总自旋 S (用于列)
    """
    # 1. 计算维度并检查
    n_rows = int(2 * s1 + 1)
    n_cols = int(2 * s2 + 1)
    
    if matrix.shape != (n_rows, n_cols):
        print(f"Warning: Matrix shape {matrix.shape} mismatch with spins S1={s1}, S2={s2} (expected {n_rows}x{n_cols})")
        return

    # 2. 生成 Sz 序列 (假设顺序是从 -S 到 +S，与你给的例子一致)
    # 使用 linspace 确保浮点数精度，例如 -1.5, -0.5, 0.5, 1.5
    row_sz_vals = np.linspace(-s1, s1, n_rows)
    col_sz_vals = np.linspace(-s2, s2, n_cols)

    # 3. 定义格式化参数
    # col_width: 每一列数据的总宽度，根据 (0.000000, 0.000000) 的长度预估，留足余量
    col_width = 25 
    # label_width: 行首标签的宽度
    label_width = 16 
    
    # 4. 打印表头 (列标签)
    # 先打印行标签占位的空白
    header = " " * label_width
    for sz in col_sz_vals:
        # 格式化 Sz 标签，居中对齐
        header += f"|Sz={sz:5.2f}>".center(col_width)
    
    print("Actual matrix elements:")
    print(header)
    print("-" * len(header)) # 打印一条分割线，更美观

    # 5. 打印每一行
    for i, row_sz in enumerate(row_sz_vals):
        # 构造行首标签: <Sz=...|
        row_str = f"<Sz={row_sz:5.2f}|".ljust(label_width)
        
        for j in range(n_cols):
            val = matrix[i, j]
            # 格式化复数: (实部, 虚部)
            # {val.real:9.6f}: 
            #   9 表示数字最小占9位(含符号和小数点)，确保对齐
            #   .6f 表示保留6位小数
            #   如果数字很小是 -0.000000，这种格式能保证宽度一致
            val_str = f"({val.real:9.6f},{val.imag:9.6f})"
            
            # 将数值字符串在列宽内居中
            row_str += val_str.center(col_width)
        
        print(row_str)

def analyze_soc(td, soctype='SOMF'):
    '''
    td: an instance of pyscf.sftda.TDA_SF
    '''

    if soctype == 'SOMF':
        soc_ao = calc_ao_soc_1e(td._scf.mol, Z='one')
        soc_ao += calc_ao_soc_2e(td._scf)
    elif soctype == 'Zeff':
        soc_ao = calc_ao_soc_1e(td._scf.mol, Z='orca')
    elif soctype == '1e':
        soc_ao = calc_ao_soc_1e(td._scf.mol, Z='one')
    else:
        raise ValueError(f"soctype={soctype} is not supported.")

    for ket in range(td.nstates):
        for bra in range(ket+1, td.nstates):
            print(f'SOC Matrix Element between State {bra+1} and State {ket+1}:')
            s21 = spin_square(td._scf, td.xy[bra], extype=td.extype, tdtype='TDA')
            s22 = spin_square(td._scf, td.xy[ket], extype=td.extype, tdtype='TDA')
            s1 = round(-1 + np.sqrt(1 + 4 * s21)) / 2
            s2 = round(-1 + np.sqrt(1 + 4 * s22)) / 2
            print(f'Spin S values: Bra State {bra+1}: S={s1}({s21:.4f}), Ket State {ket+1}: S={s2}({s22:.4f})')
            soc_mat = calc_soc(soc_ao, td._scf, td.xy[bra], td.xy[ket], s1, s2)
            print_pretty_soc(soc_mat * HARTREE2WAVENUMBER, s1, s2)
            print(f'SOCC = {np.linalg.norm(soc_mat) * HARTREE2WAVENUMBER}')
            print()

def build_and_diagonalize_soc(td, soctype='SOMF', verbose=True):
    '''
    构建完整的SOC哈密顿量并对角化
    td: pyscf.sftda.TDA_SF 实例
    '''
    
    # 1. 准备 SOC 积分
    if soctype == 'SOMF':
        soc_ao = calc_ao_soc_1e(td._scf.mol, Z='one')
        soc_ao += calc_ao_soc_2e(td._scf)
    elif soctype == 'Zeff':
        soc_ao = calc_ao_soc_1e(td._scf.mol, Z='orca')
    elif soctype == '1e':
        soc_ao = calc_ao_soc_1e(td._scf.mol, Z='one')
    else:
        raise ValueError(f"soctype={soctype} is not supported.")

    nstates = td.nstates
    
    # 2. 预计算每个态的自旋 S 和在总矩阵中的索引偏移量
    state_info = [] # 存储 (S值, start_index, end_index, energy)
    current_idx = 0
    
    print("Pre-calculating state spins and dimensions...")
    for i in range(nstates):
        # 计算 S^2 和 S
        # 注意：这里假设 spin_square 是你已有的函数
        s2_val = spin_square(td._scf, td.xy[i], extype=td.extype, tdtype='TDA')
        s_val = round(-1 + np.sqrt(1 + 4 * s2_val)) / 2
        
        # 确定该态的子空间维度 (2S+1)
        dim = int(2 * s_val + 1)
        
        # 获取该态的能量 (Hartree)
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
    print(f"Total Hamiltonian dimension: {total_dim} x {total_dim}")
    
    # 3. 初始化大哈密顿量 (复数矩阵)
    # 单位：Hartree
    H_soc = np.zeros((total_dim, total_dim), dtype=np.complex128)
    
    # 4. 填充矩阵
    # 4.1 填充对角块 (零级能量)
    for info in state_info:
        # 对角线上加上激发能 E_I * Identity
        # 注意：这里我们只加能量，如果同态内部有SOC分裂(如零场分裂)，
        # 需要计算 calc_soc(..., state_i, state_i, ...) 并加到对角块。
        # 通常简单处理时，只考虑态间耦合，对角块仅为能量。
        idx_slice = slice(info['start'], info['end'])
        H_soc[idx_slice, idx_slice] = np.eye(info['dim']) * info['energy']

    # 4.2 填充非对角块 (SOC 耦合)
    # 你的 calc_soc 假设 bra 和 ket 不同。
    for i in range(nstates):
        for j in range(i + 1, nstates):
            bra_info = state_info[j] # 行索引对应 Bra
            ket_info = state_info[i] # 列索引对应 Ket
            
            # 调用你写好的函数计算块矩阵
            # 注意参数顺序：(soc_ao, mf, xy_bra, xy_ket, s_bra, s_ket)
            # 你的函数定义是 xy1(bra), xy2(ket), s1(bra), s2(ket)
            sub_mat = calc_soc(soc_ao, td._scf, 
                               td.xy[j], td.xy[i], 
                               bra_info['s'], ket_info['s'])
            print_pretty_soc(sub_mat * HARTREE2WAVENUMBER, bra_info['s'], ket_info['s'])
            print(f'SOCC = {np.linalg.norm(sub_mat) * HARTREE2WAVENUMBER}')
            
            # 填入上三角
            H_soc[bra_info['start']:bra_info['end'], ket_info['start']:ket_info['end']] = sub_mat
            
            # 填入下三角 (厄米共轭)
            H_soc[ket_info['start']:ket_info['end'], bra_info['start']:bra_info['end']] = sub_mat.conj().T

    # 5. 对角化
    print("Diagonalizing SOC Hamiltonian...")
    eigvals, eigvecs = np.linalg.eigh(H_soc)
    
    # 6. 结果分析与打印
    if verbose:
        print("\n" + "="*60)
        print("SOC States Analysis")
        print("="*60)
        
        # 找到基态能量作为参考 (通常是 H_soc 的最小本征值，或者原始 SCF 能量)
        # 这里我们展示相对于最低 SOC 态的相对能量
        min_e = np.min(eigvals)
        
        for idx, e in enumerate(eigvals):
            # 能量转换为 cm-1
            rel_e_cm = (e - min_e) * HARTREE2WAVENUMBER
            
            print(f"SOC State {idx+1}: E = {(e-min_e)*HARTREE2EV:.6f} eV (Rel: {rel_e_cm:.2f} cm^-1)")
            
            # 分析成分
            vec = eigvecs[:, idx]
            norm_sq = np.abs(vec)**2
            
            # 找出主要贡献的无SOC态
            # 我们需要把系数按照态进行归并 (sum over Ms)
            print("  Composition:")
            for info in state_info:
                # 该电子态所有 Ms 分量的概率和
                prob = np.sum(norm_sq[info['start']:info['end']])
                if prob > 0.01: # 只打印贡献大于 1% 的态
                    print(f"    {prob*100:5.1f}% from Scalar State {info['state_id']+1} (S={info['s']})")
            print("-" * 30)

    return eigvals, eigvecs, H_soc