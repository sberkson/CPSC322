import os

from mypytable import MyPyTable

def main():
    mpg_fname = os.path.join("input_data", "auto-mpg.txt")
    prices_fname = os.path.join("input_data", "auto-prices.txt")
    print(mpg_fname, prices_fname)

if __name__ == "__main__":
    main()

def print_line():
    '''Prints a line of dashes
    '''
    print("-" * 20)

# Task 1 - I kind of manipulate as I go, first taking care of grabbing initial
# stats for the first table, and then dropping the rows and saving to output file when 
# done.  I then move on to the second table, computing stats and then dropping the rows 
# after I have finished computed the necessary stats, ultimately saving to the correct
# output file.
def task1():
    print_line()
    print("Task 1")
    print_line()
    table = MyPyTable()
    table.load_from_file("input_data/auto-mpg.txt")
    header = table.get_column_names()

    print("auto-mpg.txt")
    print_line()
    rows, columns = table.get_shape()
    print("No. of instances: ", (rows + 1)) # Add one to account for 0 index

    duplicates = table.find_duplicates(header)
    print("Duplicates: ", duplicates)

    print_line()
    print("auto-prices.txt")
    print_line()

    table.drop_rows(duplicates)
    table.save_to_file("output_data/auto-mpg.txt")
    rows, columns = table.get_shape()


    table2 = MyPyTable()
    table2.load_from_file("input_data/auto-prices.txt")
    header2 = table2.get_column_names()
    row2, column2 = table2.get_shape()
    print("No. of instances: ", (row2 + 1))

    duplicates2 = table2.find_duplicates(header2)
    print("Duplicates: ", duplicates2, end="\n")
    table2.drop_rows(duplicates2)
    table2.save_to_file("output_data/auto-prices.txt")

    print_line()
    print("duplicates removed (saved auto-mpg-nodups.txt)")
    print_line()

    row, column = table.get_shape()
    print("No. of instances: ", (row + 1))

    duplicates = table.find_duplicates(header)
    print("Duplicates: ", duplicates)

    print_line()
    print("duplicates removed (saved auto-prices-nodups.txt)")
    print_line()

    row2, column2 = table2.get_shape()
    print("No. of instances: ", (row2 + 1))

    duplicates2 = table2.find_duplicates(header2)
    print("Duplicates: ", duplicates2)
    header.append(header2[-1])
    return table, table2, header

# For this task, I opened up both text files side to side and went item over item, using 
# command + f to search and ensure that every instance was valid.
def task2():
    print_line()
    print("Task 2:")
    print_line()
    table = MyPyTable()
    table.load_from_file("input_data/auto-mpg-clean.txt")

def task3(table1, table2, header):
    print_line()
    print("Task 3")
    print_line()
    print("combined table (saved as auto-data.txt): ")
    print_line()

    table1.perform_full_outer_join(table2, "model")
    row, column = table1.get_shape()
    duplicates = table1.find_duplicates(header[:-1])

    print("No. of instances: ", (row + 1))
    print("Duplicates: ", duplicates)
    table1.save_to_file("output_data/auto-data.txt")

def task4():
    print_line()
    print("Task 4:")
    print_line()

    #stats = MyPyTable()
    table = MyPyTable()
    table.load_from_file("output_data/auto-data.txt")
    table.convert_to_numeric()
    table.remove_rows_with_missing_values()
    stats = table.compute_summary_statistics(["mpg", "displacement", "horsepower", "weight", "acceleration", "model year", "msrp"])

# Task 5
def task5():
    print_line()
    print("Task 5:")
    print_line()

table1, table2, header = task1()
task2()
task3(table1, table2, header)
#task4()








