import os
import csv
import random

def generate_dataset():
    # define the path to the folder containing the CSV files
    folder_path = "tweets_w_emojis"
    
    # define the list to store the selected tweets
    selected_tweets = []
    
    # loop through each CSV file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)
                # skip the header row if there is one
                next(csv_reader, None)
                for row in csv_reader:
                    if row != []:           
                        selected_tweets.append(row)
                
    # randomly select 200 tweets from the list of selected tweets
    random.shuffle(selected_tweets)
    selected_tweets = selected_tweets[:200]

    # export the selected tweets to csv
    with open('data.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(selected_tweets)


if __name__ == "__main__":
    generate_dataset()