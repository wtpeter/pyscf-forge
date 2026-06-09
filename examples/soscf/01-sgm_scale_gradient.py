from pyscf import gto
from pyscf.soscf.sgm import SGM

atom = '''
O 0.00000000 -0.00000000 1.20431663
C 0.00000000 0.00000000 0.00465945
H 0.00000000 0.93457758 -0.58958804
H 0.00000000 -0.93457758 -0.58958804
'''
mol = gto.M(atom=atom, charge=0, spin=0, basis='cc-pvtz', symmetry=True)
mf0 = mol.ROKS(xc='BHandHLYP')
mf0.kernel()

setocc = mf0.to_uks().mo_occ
setocc[1][0] -= 1 
setocc[0][8] += 1  
print(setocc)
ro_occ = setocc[0] + setocc[1]
print(ro_occ)

mol.spin = 2
mf4 = mol.ROKS(xc='BHandHLYP')
mf4.mo_coeff = mf0.mo_coeff
mf4.mo_occ = ro_occ
sgm_mf = SGM(mf4)
sgm_mf.kernel()

mf5 = mol.ROKS(xc='BHandHLYP')
mf5.mo_coeff = mf0.mo_coeff
mf5.mo_occ = ro_occ
sgm_mf1 = SGM(mf5)
sgm_mf1.gradient_scale = 0.5
sgm_mf1.kernel() # -94.967014
