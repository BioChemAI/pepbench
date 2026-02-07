'''
for rf/svm/xgb
fun:aas--->descriptors
dim:48, numpy array
'''
import numpy as np
from peptidy import descriptors

class PeptidyDescriptorEncoder:
    def __init__(self):
        self.aas = ['freq_A', 'freq_C', 'freq_D', 'freq_E', 'freq_F', 'freq_G',
                    'freq_H', 'freq_I', 'freq_K', 'freq_L', 'freq_M', 'freq_N',
                    'freq_P', 'freq_Q', 'freq_R', 'freq_S', 'freq_T', 'freq_V',
                    'freq_W', 'freq_Y', 'freq_R_m', 'freq_R_d', 'freq_C_m',
                    'freq_K_a', 'freq_Y_p', 'freq_R_s', 'freq_S_p', 'freq_T_p']
        self.elements = ['n_C', 'n_H', 'n_N', 'n_O', 'n_S', 'n_P']


    def encode(self,sequence):
        # sequence = sequence.upper()
        if isinstance(sequence, np.ndarray):
            sequence = sequence.tolist()

        n_samples = len(sequence)
        feature_list = []

        for seq in sequence:
            if not isinstance(seq, str) or len(seq) == 0:
                raise ValueError(f"Invalid sequence: {seq}")
            
            feature_vector = []
            ali_index = descriptors.aliphatic_index(seq)  # the aliphatic index of a peptide
            feature_vector.append(ali_index)

            ami_frequencies = descriptors.aminoacid_frequencies(seq)  # the frequency of all amino acids in the input sequence
            for aa in self.aas:
                feature_vector.append(ami_frequencies[aa])

            aroma = descriptors.aromaticity(seq)  # the sum of the frequencies of aromatic amino-acids
            feature_vector.append(aroma)

            aver_n_rotatable_bonds = descriptors.average_n_rotatable_bonds(seq)  # the number of total rotatable bonds divided by the number of amino acids in the peptide
            feature_vector.append(aver_n_rotatable_bonds)

            charg = descriptors.charge(seq)  # the total charge of the sequence
            feature_vector.append(charg)

            charg_density = descriptors.charge_density(seq)  # the charge of the peptide normalized by weight
            feature_vector.append(charg_density)

            hydro_aa_ratio = descriptors.hydrophobic_aa_ratio(seq)  # the total ratio of hydrophobic amino-acids (A, C, C_m, F, I, L, M, and V) in a peptide
            feature_vector.append(hydro_aa_ratio)

            instab_index = descriptors.instability_index(seq)  # the instability index of the peptide
            feature_vector.append(instab_index)

            isoe_point = descriptors.isoelectric_point(seq)  # the isoelectric point of the peptide
            feature_vector.append(isoe_point)

            seq_len = descriptors.length(seq)  # the length of peptide
            feature_vector.append(seq_len)

            mole_formula = descriptors.molecular_formula(seq)  # the closed molecular formula of the amino acid sequence of the peptide
            for formula in self.elements:
                feature_vector.append(mole_formula[formula])

            mole_weight = descriptors.molecular_weight(seq)  # the weight (g/mol) of the peptide without peptide bonds
            feature_vector.append(mole_weight)

            n_h_accept = descriptors.n_h_acceptors(seq)  # the total number of hydrogen bond acceptors in the peptide
            feature_vector.append(n_h_accept)

            n_h_don = descriptors.n_h_donors(seq)  # the total number of hydrogen bond donors in the peptide
            feature_vector.append(n_h_don)

            topo_polar_surface_area = descriptors.topological_polar_surface_area(seq)  # the total topological polar surface area of the peptide
            feature_vector.append(topo_polar_surface_area)

            x_logp_ener = descriptors.x_logp_energy(seq)  # the sum of xlogP index of the peptide divided by the length of the peptide
            feature_vector.append(x_logp_ener)  # (1,48)

            feature_list.append(feature_vector)  # (n_samples, 48)
    

        return np.array(feature_list, dtype=np.float32)
