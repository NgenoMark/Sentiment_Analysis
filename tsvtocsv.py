import glob
import pandas as pd

for tsv_file in glob.glob('*.tsv'):
    df = pd.read_csv(tsv_file, sep='\t')
    df.to_csv(tsv_file.replace('.tsv', '.csv'), index=False)


# code to combine data files
# import pandas as pd
#
# # Load each CSV
# df1 = pd.read_csv('file1.csv')
# df2 = pd.read_csv('file2.csv')
# df3 = pd.read_csv('file3.csv')
#
# # Combine them
# combined_df = pd.concat([df1, df2, df3], ignore_index=True)
#
# # Save to a new CSV
# combined_df.to_csv('combined.csv', index=False)
