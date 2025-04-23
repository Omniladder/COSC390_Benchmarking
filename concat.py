import pandas as pd

df = pd.DataFrame({"Column 1:" : [1,2], "Column 2:" : [2,3]})
df_2 = pd.DataFrame({"Column 1:" : [4,5], "Column 2:" : [6,7]})
df_3 = pd.DataFrame({"Column 1:" : [8,9], "Column 2:" : [10,11]})
combined_df = pd.DataFrame({"Column 1:" : [], "Column 2:" : []})


combined_df = pd.concat([combined_df, df])
combined_df = pd.concat([combined_df, df_2])
combined_df = pd.concat([combined_df, df_3])
combined_df.reset_index(drop=True, inplace=True)


print(combined_df)
