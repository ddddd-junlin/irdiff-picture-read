# IRDIFF 采样过程实现

这是根据 Algorithm 2 (IRDIFF Sampling Procedure) 伪代码实现的 Python 脚本。

## 功能说明

该脚本实现了 IRDIFF 算法的完整采样流程，用于生成与蛋白质结合位点结合的配体分子。

## 主要组件

1. **ProteinBindingSite**: 蛋白质结合位点数据结构
2. **LigandMolecule**: 配体分子数据结构
3. **IRDIFFModel**: IRDIFF 学习模型 φθ
4. **PMINet**: 预训练的 PMINet 模型
5. **ExternalDatabase**: 外部数据库 D 的模拟实现
6. **IRDIFFSampler**: 主采样器类，实现完整的 Algorithm 2 流程

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

直接运行脚本：

```bash
python irdiff_sampling.py
```

## 算法步骤对应

脚本中的方法对应伪代码中的各个步骤：

- `sample_num_atoms()` - 步骤1: 采样配体原子数 NM
- `move_com_to_zero()` - 步骤2: 移动蛋白质质心到零
- `sample_initial_ligand()` - 步骤3: 采样初始配体坐标和类型
- `embed_protein()` - 步骤5: 嵌入蛋白质特征
- `embed_ligand()` - 步骤7: 嵌入配体特征
- `self_augmentation()` - 步骤8: 自增强 (Eq. 10)
- `database_augmentation()` - 步骤9: 数据库增强 (Eq. 12)
- `predict_denoised()` - 步骤10: 预测去噪结果 (Eqs. 14, 15)
- `sample_from_posterior()` - 步骤11: 从后验采样 (Eq. 3)
- `sample()` - 主采样函数，实现完整的循环流程

## 注意事项

1. 这是一个框架实现，实际的模型架构和参数需要根据论文中的具体描述进行调整
2. 数据库查询功能目前是模拟实现，实际使用时需要连接真实的配体数据库
3. 扩散过程的噪声调度和采样策略可能需要根据具体需求优化
4. 如果有 GPU，可以在 `IRDIFFSampler` 初始化时将 `device` 参数改为 `'cuda'`

## 自定义使用

```python
from irdiff_sampling import IRDIFFSampler, IRDIFFModel, PMINet, ExternalDatabase, ProteinBindingSite

# 初始化模型和数据库
model = IRDIFFModel()
pminet = PMINet()
database = ExternalDatabase()

# 创建采样器
sampler = IRDIFFSampler(model, pminet, database, device='cpu')

# 准备蛋白质结合位点
protein = ProteinBindingSite(
    coordinates=your_protein_coords,  # numpy array [N, 3]
    atom_types=your_protein_types      # numpy array [N]
)

# 执行采样
ligand = sampler.sample(protein, k=5, T=1000)
```

