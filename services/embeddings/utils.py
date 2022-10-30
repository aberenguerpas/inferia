import pandas as pd

def dictToDataframe(data):
    
    try:
        return pd.DataFrame.from_dict(data)
    except Exception as e:
        print('Error conveting dic to dataframe')
        print(e)
        return 0
