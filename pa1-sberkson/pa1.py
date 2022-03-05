# Sam Berkson
# CPSC 322
# PA 1
# Jan 26, 2022
import os
import csv

# one required function that is tested with the provided unit test in test_pa1.py
# do not modify this function header or the unit test
def remove_missing_values(table, header, col_name):
    """Makes a new table that is a copy of the table parameter (without modifying the original
    table), but with rows removed from the table that have missing values in the column with
    label col_name

    Args:
        table (list of list): Data in 2D table format
        header (list of str): Column names corresponding to the table (in the same order)
        col_name (str): Represents the name of the column to check for missing values

    Returns:
        list of list: The new table with removed rows
    """

    # Create a new table that is a copy of the table parameter
    new_table = []
    for row in table:
        new_table.append(row)
    
    # Create a list of indices of the rows to remove
    indices_to_remove = []
    for i in range(len(new_table)):
        if new_table[i][header.index(col_name)] == "":
            indices_to_remove.append(i)
    
    # Remove the rows from the new table
    for i in reversed(indices_to_remove):
        del new_table[i]
    
    return new_table

# Function to load file into 2D list.  Accepts filename as parameter, returns header list and table 2D list
def load_file(filename):
    file = open(filename, 'r')
    reader = csv.reader(file)

    header = []
    header = next(reader)

    table = []
    for row in reader:
        table.append(row)

    return header, table
    file.close()

# Function to determine the most common occuring value in the IMDb column of the table
def highest_imdb_rating(table, header):
    return max(table, key=lambda x: x[header.index("IMDb")])[header.index("IMDb")]
    # I know this line is too fancy, I watched a cool youtube video over break about lambda functions and I just wanted to see if it works.

# Function to determine which streaming service hosts the most shows
def biggest_streaming_service(table, header):
    # Finds sum of streaming services 
    netflix_sum = 0
    hulu_sum = 0
    amazon_prime_sum = 0 
    disney_sum = 0

    for row in table:
        netflix_sum += int(row[header.index("Netflix")])
        hulu_sum += int(row[header.index("Hulu")])
        amazon_prime_sum += int(row[header.index("Prime Video")])
        disney_sum += int(row[header.index("Disney+")])

    # Determines and returns which streaming service has the most shows
    max_sum = max(netflix_sum, hulu_sum, amazon_prime_sum, disney_sum)

    if max_sum == netflix_sum:
        return "Netflix"
    elif max_sum == hulu_sum:
        return "Hulu"
    elif max_sum == amazon_prime_sum:
        return "Prime Video"
    else:
        return "Disney+"

# Function to find most commonly occuring value in IMDb column
def most_common_imdb_rating(table, header):
    table = remove_missing_values(table, header, header[4])
    imdb_list = []
    for row in table:
        imdb_list.append(row[header.index("IMDb")])
    return max(set(imdb_list), key=imdb_list.count)
   
def main(): 
    """
    Drives the program
    """
    filename = os.path.join("input_data", "tv_shows.csv")

    header, table = load_file(filename)
    for item in header:
        remove_missing_values(table, header, item)
    # Q1
    print("The highest IMDB rating is: " + highest_imdb_rating(table, header))
    service = biggest_streaming_service(table, header)
    # Q2
    print("The streaming service hosting the most TV shows is: " + service)
    # Q3
    print("The most common IMDb rating is: " + most_common_imdb_rating(table, header))

if __name__ == "__main__":
    main()
