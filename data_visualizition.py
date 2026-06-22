# %%[1]
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from matplotlib.ticker import PercentFormatter
# %%[2  Data for Accuracy per Test Case ]
df_sheet1 = pd.read_excel('Custom LLM Accuracy.xlsx',sheet_name='Sheet1', header=None)
df_sheet2 = pd.read_excel('Custom LLM Accuracy.xlsx',sheet_name='Sheet2', header=None)


Accuracy_Custom_LLM = df_sheet1.iloc[28, 1:11].values
Accuracy_Original_LLM = df_sheet1.iloc[28, 12:22].values
Accuracy_Zero_Shot_LLM = df_sheet2.iloc[28, 1:11].values
Accuracy_Without_CoT_LLM = df_sheet2.iloc[28, 38:48].values
Accuracy_1_Shot_Calculation_LLM = df_sheet2.iloc[28, 55:65].values
Accuracy_1_Shot_definition_LLM = df_sheet2.iloc[28, 67:77].values
Accuracy_2_Shot_LLM = df_sheet2.iloc[28, 81:91].values
Accuracy_5_Shot_LLM = df_sheet2.iloc[28, 93:103].values
Accuracy_4_Shot_LLM = df_sheet2.iloc[28, 106:116].values
Accuracy_6_Shot_LLM = df_sheet2.iloc[28, 119:129].values
Accuracy_8_Shot_LLM = df_sheet2.iloc[28, 132:142].values
Accuracy_10_Shot_LLM = df_sheet2.iloc[28, 145:155].values

print(f"Custom LLM data: {Accuracy_Custom_LLM}")
print(f"Original LLM data: {Accuracy_Original_LLM}")
print(f"Zero-Shot LLM data: {Accuracy_Zero_Shot_LLM}")
print(f"Without CoT LLM data: {Accuracy_Without_CoT_LLM}")
print(f"One-Shot Calculation LLM data: {Accuracy_1_Shot_Calculation_LLM}")
print(f"One-Shot Definition LLM data: {Accuracy_1_Shot_definition_LLM}")
print(f"Two-Shot Without CoT LLM data: {Accuracy_2_Shot_LLM}")
print(f"Five-Shot Without CoT LLM data: {Accuracy_5_Shot_LLM}")
print(f"Four-Shot Without CoT LLM data: {Accuracy_4_Shot_LLM}")
print(f"Six-Shot Without CoT LLM data: {Accuracy_6_Shot_LLM}")
print(f"Eight-Shot Without CoT LLM data: {Accuracy_8_Shot_LLM}")
print(f"Ten-Shot Without CoT LLM data: {Accuracy_10_Shot_LLM}")


# %%[3  Accuracy Comparison between Custom LLM and Original LLM ]

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

plot_data = pd.DataFrame({
    'Accuracy': list(Accuracy_Custom_LLM) + list(Accuracy_Original_LLM),
    'Model Type': ['Custom LLM \n（with 15 Few-Shot examples）'] * len(Accuracy_Custom_LLM) + ['Original LLM'] * len(Accuracy_Original_LLM)
})
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='Model Type', y='Accuracy', data=plot_data, palette="Set1", medianprops={'color': 'yellow', 'linewidth': 2}, whiskerprops={'color': 'blue', 'linestyle': '--'}, capprops={'color': 'blue', 'linewidth': 4}, width=0.6)
#sns.swarmplot(x='Model Type', y='Accuracy', data=plot_data, color=".25")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
#plt.title('Accuracy Comparison between Custom LLM and Original LLM', fontsize=15)
plt.xlabel('Model Type', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.8, 1.0)
plt.yticks([0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0])
plt.tight_layout()
plt.show()


# %%[4  Accuracy Comparison between Zero-Shot LLM and Without CoT LLM ]

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

plot_data = pd.DataFrame({
    'Accuracy': list(Accuracy_Custom_LLM) + list(Accuracy_Zero_Shot_LLM) + list(Accuracy_Without_CoT_LLM),
    'Model Type': ['Custom LLM \n（with 15 Few-Shot examples）'] * len(Accuracy_Custom_LLM) + ['Zero-Shot LLM'] * len(Accuracy_Zero_Shot_LLM) + ['With Role-Play only LLM'] * len(Accuracy_Without_CoT_LLM)
})
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='Model Type', y='Accuracy', data=plot_data, palette="Set1", medianprops={'color': 'yellow', 'linewidth': 2}, whiskerprops={'color': 'blue', 'linestyle': '--'}, capprops={'color': 'blue', 'linewidth': 4}, width=0.6)
#sns.swarmplot(x='Model Type', y='Accuracy', data=plot_data, color=".25")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
#plt.title('Accuracy Comparison between Custom LLM, Zero-Shot LLM, Without CoT and Few-Shot LLM', fontsize=15)
plt.xlabel('Model Type', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.8, 1.0)
plt.yticks([0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0])
plt.tight_layout()
plt.show()


# %% [5 Accuracy Comparison between different types of one-shot and zero-shot LLM]

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

plot_data = pd.DataFrame({
    'Accuracy': list(Accuracy_1_Shot_Calculation_LLM) + list(Accuracy_1_Shot_definition_LLM) + list(Accuracy_Zero_Shot_LLM),
    'Model Type': ['One-Shot LLM (Calculation)'] * len(Accuracy_1_Shot_Calculation_LLM) + ['One-Shot LLM (Definition)'] * len(Accuracy_1_Shot_definition_LLM) + ['Zero-Shot LLM'] * len(Accuracy_Zero_Shot_LLM)
})
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='Model Type', y='Accuracy', data=plot_data, palette="Set1", medianprops={'color': 'yellow', 'linewidth': 2}, whiskerprops={'color': 'blue', 'linestyle': '--'}, capprops={'color': 'blue', 'linewidth': 4}, width=0.6)
#sns.swarmplot(x='Model Type', y='Accuracy', data=plot_data, color=".25")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
#plt.title('Accuracy Comparison between different types of One-Shot LLM and Zero-Shot LLM', fontsize=15)
plt.xlabel('Model Type', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.8, 1.0)
plt.yticks([0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0])
plt.tight_layout()
plt.show()

# %%[6]  Accuracy Comparison between different amount of shot examples  LLM and Zero-Shot LLM]

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

plot_data = pd.DataFrame({
    'Accuracy': list(Accuracy_2_Shot_LLM) + list(Accuracy_4_Shot_LLM) + list(Accuracy_8_Shot_LLM) + list(Accuracy_Zero_Shot_LLM),
    'Model Type': ['2-Shot LLM'] * len(Accuracy_2_Shot_LLM) + ['4-Shot LLM'] * len(Accuracy_4_Shot_LLM) + ['8-Shot LLM'] * len(Accuracy_8_Shot_LLM) + ['Zero-Shot LLM'] * len(Accuracy_Zero_Shot_LLM)
})
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='Model Type', y='Accuracy', data=plot_data, palette="Set1", medianprops={'color': 'yellow', 'linewidth': 2}, whiskerprops={'color': 'blue', 'linestyle': '--'}, capprops={'color': 'blue', 'linewidth': 4}, width=0.6)
#sns.swarmplot(x='Model Type', y='Accuracy', data=plot_data, color=".25")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
#plt.title('Accuracy Comparison between different Few-Shot sizes LLM and Zero-Shot LLM', fontsize=13)
plt.xlabel('Model Type', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.8, 1.0)
plt.yticks([0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0])
plt.tight_layout()
plt.show()




