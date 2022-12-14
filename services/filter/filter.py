import argparse
import glob
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser(description='Filter dataset folder.')
    parser.add_argument('-i', '--input', required=True,
                        help='Name of the input folder storing the CSV')
    parser.add_argument('-o', '--output', required=True,
                        help='Name of the output folder to save the filtered tables')
    args = parser.parse_args()

    list_files = glob.glob(os.path.join(args.input, '*.csv'))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    discarted = 0

    try:
        
        for file in list_files:
            file_name = file.split("/")[-1]
            df = pd.read_csv(file, encoding = 'ISO-8859-1', on_bad_lines='skip')

            if len(df.index)>1000 and len(df.index)<1000000:
                df.to_csv(args.output+"/"+file_name, index=False)
            else:
                discarted+=1
            
    except Exception as e:
        print(e)

    print("Tables discarted:", discarted)

if __name__ == '__main__':
    main()