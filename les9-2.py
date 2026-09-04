# 第一步：解决中文乱码+后端兼容
import matplotlib

matplotlib.use('TkAgg')  # 适配PyCharm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 中文配置
plt.rcParams['axes.unicode_minus'] = False  # 负号显示

# 导入核心库
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold  # 特征选择
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                             confusion_matrix, roc_curve, auc, classification_report)
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# -------------------------- 1. 数据读取（训练集+测试集） --------------------------
# 1.1 训练集：读取前30000条（用于生成4:6分布的训练数据）
train_df = pd.read_csv('train.csv', nrows=30000)
print(f"=== 训练集原始信息 ===")
print(f"训练集读取行数: {len(train_df)} 行")
print(f"训练集原始类别分布:\n{train_df['Class'].value_counts()}")
train_fraud_ratio = train_df['Class'].value_counts()[1] / len(train_df) * 100
print(f"训练集欺诈样本占比: {train_fraud_ratio:.4f}%")

# 1.2 测试集：读取接下来的10000条（跳过前30000条，读取10000条）
test_df = pd.read_csv('train.csv', skiprows=30000, nrows=10000, header=0)
# 重置列名（避免skiprows导致列名丢失）
test_df.columns = train_df.columns
print(f"\n=== 测试集原始信息 ===")
print(f"测试集读取行数: {len(test_df)} 行")
print(f"测试集原始类别分布:\n{test_df['Class'].value_counts()}")
test_fraud_ratio = test_df['Class'].value_counts()[1] / len(test_df) * 100
print(f"测试集欺诈样本占比: {test_fraud_ratio:.4f}%")


# -------------------------- 2. 缺失值处理（训练集+测试集） --------------------------
def process_missing(data, is_train=True, imputer=None):
    """
    缺失值处理函数：训练集拟合imputer，测试集复用
    """
    missing_values = data.isnull().sum()
    missing_total = missing_values.sum()
    if is_train:
        print(f"\n=== 训练集缺失值统计 ===")
        print(f"总缺失值数量: {missing_total}")
        print(f"各列缺失值分布:\n{missing_values[missing_values > 0]}")

        if missing_total > 0:
            X_temp = data.drop(['id', 'Class'], axis=1)
            y_temp = data['Class']
            id_temp = data['id']

            # 训练集拟合imputer
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X_temp)

            # 重构DataFrame
            df_imputed = pd.DataFrame(X_imputed, columns=X_temp.columns)
            df_imputed['id'] = id_temp.values
            df_imputed['Class'] = y_temp.values
            data = df_imputed
            print(f"训练集缺失值处理完成：均值填充")

            # 校验无NaN
            post_missing = data.drop(['id', 'Class'], axis=1).isnull().sum().sum()
            if post_missing > 0:
                raise ValueError(f"训练集缺失值处理不彻底，仍有{post_missing}个NaN！")
        else:
            print(f"训练集无缺失值")
        return data, imputer
    else:
        # 测试集复用训练集的imputer
        if missing_total > 0 and imputer is not None:
            X_temp = data.drop(['id', 'Class'], axis=1)
            y_temp = data['Class']
            id_temp = data['id']

            X_imputed = imputer.transform(X_temp)  # 仅transform，不fit
            df_imputed = pd.DataFrame(X_imputed, columns=X_temp.columns)
            df_imputed['id'] = id_temp.values
            df_imputed['Class'] = y_temp.values
            data = df_imputed
            print(f"测试集缺失值处理完成：复用训练集均值填充")
        return data


# 处理训练集缺失值
train_df, imputer = process_missing(train_df, is_train=True)
# 处理测试集缺失值（复用训练集的imputer）
test_df = process_missing(test_df, is_train=False, imputer=imputer)

# -------------------------- 3. 训练集：生成4:6分布的平衡数据 --------------------------
# 分离训练集0/1类
train_0 = train_df[train_df['Class'] == 0].reset_index(drop=True)
train_1 = train_df[train_df['Class'] == 1].reset_index(drop=True)

if len(train_1) == 0:
    raise ValueError("训练集中无欺诈样本，请扩大训练集读取行数！")

# 强制4:6分布（总10000，0类4000，1类6000）
total_target = 30000
n0_target = 12000
n1_target = 18000

# 安全校验
if len(train_0) < n0_target:
    raise ValueError(f"训练集未欺诈样本不足{n0_target}，仅{len(train_0)}个！")

# 欠采样0类到4000
train_0_sampled = train_0.sample(n=n0_target, random_state=42, replace=False)
# 合并欠采样0类+原始1类
train_temp = pd.concat([train_0_sampled, train_1], axis=0).reset_index(drop=True)
X_train_temp = train_temp.drop(['id', 'Class'], axis=1)
y_train_temp = train_temp['Class']

# SMOTE过采样1类到6000
smote = SMOTE(
    sampling_strategy={1: n1_target},
    random_state=42,
    k_neighbors=min(5, len(train_1) - 1)
)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_temp, y_train_temp)

# 重构平衡训练集
train_balanced = pd.DataFrame(X_train_balanced, columns=X_train_temp.columns)
train_balanced['Class'] = y_train_balanced
train_balanced['id'] = range(len(train_balanced))

# 训练集平衡后校验
train_bal_0 = len(train_balanced[train_balanced['Class'] == 0])
train_bal_1 = len(train_balanced[train_balanced['Class'] == 1])
print(f"\n=== 训练集平衡后分布 ===")
print(f"平衡训练集总样本: {len(train_balanced)}")
print(f"0类数量: {train_bal_0} ({train_bal_0 / len(train_balanced) * 100:.1f}%)")
print(f"1类数量: {train_bal_1} ({train_bal_1 / len(train_balanced) * 100:.1f}%)")

# -------------------------- 4. 预处理：训练集+测试集（标准化+特征选择+PCA） --------------------------
# 分离训练集特征/标签
X_train = train_balanced.drop(['id', 'Class'], axis=1)
y_train = train_balanced['Class']
# 分离测试集特征/标签（保留原始分布，不平衡）
X_test = test_df.drop(['id', 'Class'], axis=1)
y_test = test_df['Class']

# 4.1 标准化（训练集fit，测试集transform）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 复用训练集scaler

# 4.2 特征选择（训练集fit，测试集transform）
vt = VarianceThreshold(threshold=0.1)
X_train_selected = vt.fit_transform(X_train_scaled)
X_test_selected = vt.transform(X_test_scaled)  # 复用训练集vt
print(f"\n=== 特征选择结果 ===")
print(f"原始特征数: {X_train.shape[1]}, 筛选后: {X_train_selected.shape[1]}")

# 4.3 PCA降维（训练集fit，测试集transform）
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_selected)
X_test_pca = pca.transform(X_test_selected)  # 复用训练集pca

# 构建PCA可视化DataFrame
train_pca_df = pd.DataFrame(X_train_pca, columns=['主成分1', '主成分2'])
train_pca_df['交易类型'] = y_train.map({0: '未欺诈', 1: '欺诈'})
test_pca_df = pd.DataFrame(X_test_pca, columns=['主成分1', '主成分2'])
test_pca_df['交易类型'] = y_test.map({0: '未欺诈', 1: '欺诈'})

# -------------------------- 5. 训练SVM模型 --------------------------
# 训练高维SVM（用于评估）
svm_model = SVC(
    kernel='rbf',
    C=2.0,
    gamma='scale',
    random_state=42,
    probability=True
)
svm_model.fit(X_train_selected, y_train)

# 训练2维SVM（用于可视化边界）
svm_vis = SVC(
    kernel='rbf',
    C=5.0,
    gamma=0.1,
    random_state=42
)
svm_vis.fit(X_train_pca, y_train)

print(f"\n=== SVM模型训练完成 ===")
print(f"模型核函数: {svm_model.kernel}, C={svm_model.C}")

# -------------------------- 6. 测试集评估 --------------------------
# 测试集预测
y_test_pred = svm_model.predict(X_test_selected)
y_test_proba = svm_model.predict_proba(X_test_selected)[:, 1]

# 计算测试集指标
test_accuracy = accuracy_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)
test_roc_auc = auc(test_fpr, test_tpr)

print(f"\n=== 测试集（下10000条）评估结果 ===")
print(f"整体准确率: {test_accuracy:.4f}")
print(f"欺诈样本召回率: {test_recall:.4f}")
print(f"精确率: {test_precision:.4f}")
print(f"F1分数: {test_f1:.4f}")
print(f"ROC-AUC分数: {test_roc_auc:.4f}")
print("\n测试集分类报告:")
print(classification_report(y_test, y_test_pred, target_names=['未欺诈', '欺诈']))

# -------------------------- 7. 可视化（训练集+测试集） --------------------------
# 7.1 训练集vs测试集分布对比
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# 左：训练集平衡分布
ax1 = axes[0]
sns.countplot(x='Class', data=train_balanced, ax=ax1)
ax1.set_title(f'训练集平衡分布（4:6）', fontsize=12)
ax1.set_xlabel('交易类型 (0=未欺诈, 1=欺诈)')
ax1.set_ylabel('样本数量')
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height())}', (p.get_x() + 0.2, p.get_height() + 50), fontsize=10)
# 右：测试集原始分布
ax2 = axes[1]
sns.countplot(x='Class', data=test_df, ax=ax2)
ax2.set_title(f'测试集原始分布（下10000条）', fontsize=12)
ax2.set_xlabel('交易类型 (0=未欺诈, 1=欺诈)')
ax2.set_ylabel('样本数量')
for p in ax2.patches:
    ax2.annotate(f'{int(p.get_height())}', (p.get_x() + 0.2, p.get_height() + 50), fontsize=10)
plt.tight_layout()
plt.savefig('train_test_distribution.png', dpi=300)
plt.show()

# 7.2 测试集SVM分类边界
plt.figure(figsize=(12, 9))
# 绘制训练集的决策边界
x_min, x_max = X_train_pca[:, 0].min() - 2, X_train_pca[:, 0].max() + 2
y_min, y_max = X_train_pca[:, 1].min() - 2, X_train_pca[:, 1].max() + 2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
plt.contour(xx, yy, Z, colors='k', linewidths=1.5)

# 绘制测试集散点（抖动分离重合）
jitter = 0.03
test_0 = test_pca_df[test_pca_df['交易类型'] == '未欺诈']
test_1 = test_pca_df[test_pca_df['交易类型'] == '欺诈']
# 未欺诈
plt.scatter(
    test_0['主成分1'] + np.random.normal(0, jitter, len(test_0)),
    test_0['主成分2'] + np.random.normal(0, jitter, len(test_0)),
    s=40, alpha=0.7, marker='o', edgecolor='k', linewidth=0.3,
    c='#2E86AB', label=f'测试集未欺诈（{len(test_0)}个）'
)
# 欺诈
plt.scatter(
    test_1['主成分1'] + np.random.normal(0, jitter, len(test_1)),
    test_1['主成分2'] + np.random.normal(0, jitter, len(test_1)),
    s=60, alpha=0.7, marker='^', edgecolor='k', linewidth=0.3,
    c='#E63946', label=f'测试集欺诈（{len(test_1)}个）'
)

plt.title(f'测试集（下10000条）SVM分类边界', fontsize=15)
plt.xlabel(f'主成分1（解释方差：{pca.explained_variance_ratio_[0] * 100:.1f}%）', fontsize=12)
plt.ylabel(f'主成分2（解释方差：{pca.explained_variance_ratio_[1] * 100:.1f}%）', fontsize=12)
plt.legend(loc='upper right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('test_svm_boundary.png', dpi=300)
plt.show()

# 7.3 测试集混淆矩阵
plt.figure(figsize=(8, 6))
test_cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['未欺诈', '欺诈'],
            yticklabels=['未欺诈', '欺诈'],
            annot_kws={'size': 12})
plt.title('测试集混淆矩阵', fontsize=14)
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.tight_layout()
plt.savefig('test_confusion_matrix.png', dpi=300)
plt.show()

# 7.4 训练集vs测试集ROC曲线对比
plt.figure(figsize=(8, 6))
# 训练集ROC（可选）
y_train_pred_proba = svm_model.predict_proba(X_train_selected)[:, 1]
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_pred_proba)
train_roc_auc = auc(train_fpr, train_tpr)
plt.plot(train_fpr, train_tpr, label=f'训练集 ROC (AUC={train_roc_auc:.4f})', linewidth=2)
# 测试集ROC
plt.plot(test_fpr, test_tpr, label=f'测试集 ROC (AUC={test_roc_auc:.4f})', linewidth=2, linestyle='--')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('假阳性率(FPR)')
plt.ylabel('真阳性率(TPR)')
plt.title('训练集vs测试集ROC曲线')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('train_test_roc.png', dpi=300)
plt.show()

print("\n=== 可视化文件生成完成 ===")
print("1. train_test_distribution.png：训练/测试集分布对比")
print("2. test_svm_boundary.png：测试集分类边界")
print("3. test_confusion_matrix.png：测试集混淆矩阵")
print("4. train_test_roc.png：训练/测试集ROC对比")