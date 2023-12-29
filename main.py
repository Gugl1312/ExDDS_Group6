import numpy as np
import pandas as pd

def import_dataset(n):
    if n == 1: # Netflix dataset
        files = []
        for i in range(1, 5):
            files.append(pd.read_csv('data/Netflix/combined_data_' + str(i) + '.txt', names = ['Cust_Id', 'Rating', 'Date']))
            files[i - 1].dropna(inplace=True)
        return pd.concat(files)
    elif n == 2: # Jester dataset
        # import dataset 2
        files = []
        for i in range(1, 4):
            files.append(pd.read_excel('data/Jester/jester-data-' + str(i) + '.xls', header=None))
            files[i - 1].replace(99, np.nan, inplace=True)
        return pd.concat(files)
    elif n == 3: # Goodreads10K dataset
        return pd.read_csv('data/Goodreads10K/ratings.csv')
    else:
        print("Error: invalid dataset number")
        return None


def clusterBasedBanditAlgorithm(B, C, D): # Algorithm 1
    pass

def g_exploration(): # Algorithm 2
    pass

def test_all_datasets():
    for i in range(1, 4): # example to test all of the datasets
        data = import_dataset(i)

def main():
    print(import_dataset(3))



if __name__ == '__main__':
    main()