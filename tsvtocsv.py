# import glob
# import pandas as pd
#
# for tsv_file in glob.glob('*.tsv'):
#     df = pd.read_csv(tsv_file, sep='\t')
#     df.to_csv(tsv_file.replace('.tsv', '.csv'), index=False)

import pandas as pd

# Load each file (make sure these paths and filenames are correct)
df1 = pd.read_excel('ad1.xlsx')
df2 = pd.read_excel('an1.xlsx')
df3 = pd.read_excel('c1.xlsx')
df4 = pd.read_excel('cy1.xlsx')
df5 = pd.read_excel('di1.xlsx')
df6 = pd.read_excel('e1.xlsx')
df7 = pd.read_excel('ex1.xlsx')
df8 = pd.read_excel('hu1.xlsx')
df9 = pd.read_excel('in1.xlsx')
df10 = pd.read_excel('m1.xlsx')
df11 = pd.read_excel('no1.xlsx')
df12 = pd.read_excel('o1.xlsx')
df13 = pd.read_excel('s1.xlsx')
df14 = pd.read_excel('sh1.xlsx')
df15 = pd.read_csv('sentiment trainer.csv')

# Combine all dataframes
combined_df = pd.concat([
    df1, df2, df3, df4, df5, df6, df7,
    df8, df9, df10, df11, df12, df13, df14, df15
], ignore_index=True)

# Save to a new CSV
combined_df.to_csv('combined.csv', index=False)
