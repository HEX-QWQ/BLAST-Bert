import pandas as pd

# Read the input file
df = pd.read_csv('blast\dataset\data.txt', sep='\t')

# Group by qseqid and aggregate sseqid and pident into lists
grouped = df.groupby('qseqid').agg({
    'sseqid': lambda x: ' '.join(x),
    'pident': lambda x: ' '.join(map(str, x))
}).reset_index()

# Write to output file
with open('output.txt', 'w') as f:
    for _, row in grouped.iterrows():
        f.write(f"{row['qseqid']}\t{row['sseqid']}\t{row['pident']}\n")