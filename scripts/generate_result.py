import pandas as pd
import matplotlib.pyplot as plt
import os

data_directory = 'data'
result_directory = 'result'

if not os.path.exists(result_directory):
    os.makedirs(result_directory)

file_mapping = {
    "GPU+CPU6+Mem12": "GPU+CPU6+Mem12.csv",
    "CPU6+Mem12": "CPU6+Mem12.csv",
    "CPU4+Mem8": "CPU4+Mem8.csv",
    "CPU3+Mem6": "CPU3+Mem6.csv",
    "CPU2+Mem4": "CPU2+Mem4.csv",
    "CPU1+Mem2": "CPU1+Mem2.csv"
}

first_config_name = list(file_mapping.keys())[0]
first_filename = file_mapping[first_config_name]
first_filepath = os.path.join(data_directory, first_filename)
combined_df = pd.read_csv(first_filepath)[["num_nodes", "num_edges", "features", "m_param"]]

for config_name, filename in file_mapping.items():
    filepath = os.path.join(data_directory, filename)
    df = pd.read_csv(filepath)
    df = df[['latency_s']].rename(columns={'latency_s': config_name})
    combined_df = pd.concat([combined_df, df], axis=1)

gcn_csv_path = os.path.join(result_directory, 'rgcn.csv')
combined_df.to_csv(gcn_csv_path, index=False)
print(f"Successfully created the combined CSV file at: {gcn_csv_path}")
print("--- Combined Data Head ---")
print(combined_df.head())
print("--------------------------")

result_png_path = os.path.join(result_directory, 'result.png')

plt.style.use('default')
plt.figure(figsize=(16, 10))

for config_name in file_mapping.keys():
    plt.plot(combined_df['num_nodes'], combined_df[config_name], marker='o', linestyle='-', label=config_name)

plt.xscale('log')
plt.yscale('log')

plt.xlabel("Number of Nodes", fontsize=12)
plt.ylabel("Latency (s)", fontsize=12)
plt.title("Performance vs. Number of Nodes for Different Configurations", fontsize=16)

plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(result_png_path)
print(f"Successfully created the plot at: {result_png_path}")
