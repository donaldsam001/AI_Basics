from load_to_dataframe import load_cvs_to_dataframe
from text_preprocessing import clean_cv_text
import pandas as pf


folder_path = "/home/donaldsam/Downloads/AI_Basic/example_data"
df_cvs = load_cvs_to_dataframe(folder_path)
print(df_cvs.head())


df_cvs['clean_text'] = df_cvs['raw_text'].apply(clean_cv_text)