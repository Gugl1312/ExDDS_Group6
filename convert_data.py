import argparse
import pandas as pd
from tqdm import tqdm

def convert_netflix_file(file_name):
  file = pd.read_csv(file_name, names = ['user_id', 'Rating'], usecols = [0, 1], header=None)
  file['movie_id'] = 0
  print(file)
  movie_id = 0
  for index, row in tqdm(file.iterrows(), desc="Splitting movie IDs", total=len(file)):
    if row.isna().sum() > 0:
      movie_id = row['user_id'].split(':')[0]
    else:
      file.iloc[index, 2] = movie_id

  file.dropna(inplace=True)
  file.to_csv(file_name.split('.')[0] + "_converted.csv", index=False)
  return file

def combine_netflix_files():
  with tqdm(total=1, desc="Combining", unit="item") as pbar:
    files = []
    for i in range(1, 5):
      files.append(pd.read_csv('data/Netflix/combined_data_' + str(i) + '_converted.csv', names = ['user_id', 'Rating', 'movie_id'], usecols = [0, 1, 2], header=None))
    pd.concat(files).to_csv('data/Netflix/combined_data.csv', index=False)
    pbar.update(1)

def pivot_netflix_data():
   with tqdm(total=1, desc="Pivoting", unit="item") as pbar:
    data = pd.read_csv('data/Netflix/combined_data.csv')
    data.reset_index(inplace=True)
    pivoted = data.pivot_table(index='user_id', columns='movie_id', values='Rating')
    pivoted.reset_index(inplace=True)
    pivoted.astype('float64')
    pivoted.columns = pivoted.columns.astype(str)
    pivoted.to_csv('data/Netflix/data_final.csv', index=False)
    pbar.update(1)

def pivot_goodreads_data():
  with tqdm(total=1, desc="Pivoting", unit="item") as pbar:
    data = pd.read_csv('data/Goodreads10K/ratings.csv')
    data.reset_index(inplace=True)
    pivoted = data.pivot_table(index='user_id', columns='book_id', values='rating')
    pivoted.reset_index(inplace=True)
    pivoted.astype('float64')
    pivoted.columns = pivoted.columns.astype(str)
    pivoted.to_csv('data/Goodreads10K/ratings_final.csv', index=False)
    pbar.update(1)

def combine_jester_data():
  with tqdm(total=1, desc="Combining", unit="item") as pbar:
    files = []
    for i in range(1, 4):
        files.append(pd.read_excel('data/Jester/jester-data-' + str(i) + '.xls', header=None))
        files[i - 1].replace(99, 999, inplace=True)
    X = pd.concat(files)
    X.columns = X.columns.astype(str)
    X.rename(columns={'0': 'user_id'}, inplace=True)
    X['user_id'] = range(1, len(X) + 1)
    X.to_csv('data/Jester/jester_final.csv', index=False)
    pbar.update(1)

def main():
  parser = argparse.ArgumentParser(description="Process datasets based on provided names.")

  parser.add_argument('--dataset', type=str, help='Name of the dataset')

  args = parser.parse_args()

  if (args.dataset == "Netflix"):
    convert_netflix_file("data/Netflix/combined_data_1.txt")
    convert_netflix_file("data/Netflix/combined_data_2.txt")
    convert_netflix_file("data/Netflix/combined_data_3.txt")
    convert_netflix_file("data/Netflix/combined_data_4.txt")
    combine_netflix_files()
    pivot_netflix_data()
  elif (args.dataset == "Jester"):
    combine_jester_data()
  elif (args.dataset == "Goodreads"):
    pivot_goodreads_data()
  else:
    print("Error: invalid dataset name")
    return

if __name__ == "__main__":
  main()