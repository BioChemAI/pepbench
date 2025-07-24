from peptidy.descriptors import(
    aliphatic_index, aminoacid_frequencies, aromaticity,
    average_n_rotatable_bonds, charge, charge_density,
    compute_descriptors, hydrophobic_aa_ratio,
    instability_index, isoelectric_point, length,
    molecular_formula, molecular_weight, n_h_acceptors,
    n_h_donors, topological_polar_surface_area,
    x_logp_energy
)



class PeptidyEncoder:
    def __init__(self, pH=7.0, include_names=False):
        """
        基于peptidy库的多肽特征编码器，计算指定的理化描述符
        
        参数:
            pH: 计算电荷相关特征时的环境pH值（默认7.0）
            include_names: 是否返回特征名称列表（默认False，仅返回特征值数组）
        """
        self.pH = pH
        self.include_names = include_names
        
        # 定义需计算的描述符（名称+计算函数+是否需要pH参数）
        self.descriptors = [
            ('aliphatic_index', aliphatic_index, False),
            ('aromaticity', aromaticity, False),
            ('average_n_rotatable_bonds', average_n_rotatable_bonds, False),
            ('charge', charge, True),
            ('charge_density', charge_density, True),
            ('hydrophobic_aa_ratio', hydrophobic_aa_ratio, False),
            ('instability_index', instability_index, False),
            ('isoelectric_point', isoelectric_point, False),
            ('length', length, False),
            ('molecular_weight', molecular_weight, False),
            ('n_h_acceptors', n_h_acceptors, False),
            ('n_h_donors', n_h_donors, False),
            ('topological_polar_surface_area', topological_polar_surface_area, False),
            ('x_logp_energy', x_logp_energy, False)
        ]
        
        # 处理氨基酸频率（返回字典，需拆分为单独特征）
        self.aa_freq_prefix = 'aa_freq_'  # 特征名称前缀
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']  # 标准氨基酸
        # 补充可能的修饰氨基酸（根据peptidy支持的类型扩展）
        self.modified_aa = ['C_m', 'Y_p']  # 示例：甲基化半胱氨酸、磷酸化酪氨酸
        self.all_aa = self.amino_acids + self.modified_aa
        
        # 整合所有特征名称（基础描述符 + 氨基酸频率）
        self.feature_names = [name for name, _, _ in self.descriptors]
        self.feature_names += [f'{self.aa_freq_prefix}{aa}' for aa in self.all_aa]

    def encode(self, peptide_sequence):
        """
        计算多肽序列的特征向量
        
        参数:
            peptide_sequence: 字符串，如"AVILC_mY_p"
        
        返回:
            若include_names=True：元组 (特征值数组, 特征名称列表)
            否则：特征值数组（shape: (n_features,)）
        """
        features = []
        
        # 1. 计算基础描述符（非氨基酸频率）
        for name, func, need_pH in self.descriptors:
            try:
                if need_pH:
                    # 需传入pH参数的函数（如charge）
                    val = func(peptide_sequence, pH=self.pH)
                else:
                    # 无需pH参数的函数
                    val = func(peptide_sequence)
                features.append(val)
            except Exception as e:
                features.append(np.nan)
                print(f"计算 {name} 时出错: {e}")
        
        # 2. 计算氨基酸频率（拆分字典为单独特征）
        try:
            aa_freq_dict = aminoacid_frequencies(peptide_sequence)
            # 按预设氨基酸列表提取频率，未出现的用0填充
            for aa in self.all_aa:
                freq_key = f'freq_{aa}'  # 对应aminoacid_frequencies返回的键格式
                features.append(aa_freq_dict.get(freq_key, 0.0))
        except Exception as e:
            # 若计算失败，用NaN填充所有氨基酸频率特征
            features += [np.nan] * len(self.all_aa)
            print(f"计算氨基酸频率时出错: {e}")
        
        # 转换为numpy数组
        features_array = np.array(features, dtype=np.float32)
        
        if self.include_names:
            return features_array, self.feature_names
        else:
            return features_array