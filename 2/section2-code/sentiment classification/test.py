import pandas as pd

# File path to the Feather file
feather_file = r"C:\Users\VAIO\Desktop\DSC\Delver_HamNet_V5.1\Delver_HamNet\Dham_Image_Specifications.feather"

sfeature_1=pd.read_feather(feather_file,columns=[0,6])
print(sfeature_1.shape)