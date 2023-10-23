import pandas as pd

df = pd.read_csv('datashop/births.csv')

import pandas as pd
import numpy as np

# Create a list
new_row = ['1960-1-28', 97]

# Reshape the new row to match the column names
new_row = np.reshape(new_row, (1, 2))

# Create a new DataFrame from the new row
new_row_df = pd.DataFrame(new_row, columns=df.columns)

# Append the new row to the DataFrame
df = pd.concat([df, new_row_df], ignore_index=True)

# Print the DataFrame
print(df)