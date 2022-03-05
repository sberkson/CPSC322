# Sam Berkson
# CPSC 322
# PA4

from mysklearn import myutils
import copy
import csv

class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.data[0])

    def get_column_names(self):
        return self.column_names

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """
        col = -1

        if type(col_identifier) == int:
            col = col_identifier
        elif type(col_identifier) == str:
            col = self.column_names.index(col_identifier)
        else:
            #raise ValueError
            pass

        return [row[col] for row in self.data]

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """

        for row in range(len(self.data)):
            for col in range(len(self.data[row])):
                try: 
                    self.data[row][col] = float(self.data[row][col])
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.
        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        
        row_indexes_to_drop.sort(reverse = True)

        for row in row_indexes_to_drop:
            self.data.pop(row)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        inputFile = open(filename, 'r')
        reader = csv.reader(inputFile)
        header, table = [], []
        count = 0

        for value in reader:
            if count == 0:
                header = value
            else:
                table.append(value)
            count +=1

        inputFile.close

        self.column_names = header
        self.data = table
        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        with open(filename, 'w') as outputFile:
            writer = csv.writer(outputFile)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns
            list of int: list of indexes of duplicate rows found
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        
        key_column_indexes = []
        for col in key_column_names:
            key_column_indexes.append(self.column_names.index(col))
        new_table = []
        for row in self.data:
            new_table.append(tuple(row[i] for i in key_column_indexes))
        return [i for i, x in enumerate(new_table) if x in new_table[:i]]

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        self.data = [row for row in self.data if "NA" not in row and "N/A" not in row]

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """

        column = self.column_names.index(col_name)

        valid = [row[column] for row in self.data if row[column] != "NA" and row[column] != "N/A"]

        avg = sum(valid)/len(valid) 

        self.data = [[row[i] if i != column or row[i] != "NA" else avg for i in range(len(row))] for row in self.data]

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """

        name = ["attribute", "min", "max", "mid", "avg", "median"]
        newTable = []

        for col in col_names:
            row = []
            values = self.get_column(col,False)

            if not values == []:
                values.sort()
                row.append(col)
                row.append(min(values))
                row.append(max(values))
                row.append((max(values)+min(values))/2)
                row.append(sum(values)/len(values))

                if len(values) % 2 == 0:
                    row.append((values[int((len(values)-1)/2)]+values[int(((len(values)-1)/2)+1)])/2)
                else:
                    row.append(values[int((len(values)-1)/2)])

                newTable.append(row)

        return MyPyTable(name, newTable)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """

        table1 = []
        table2 = []
        headers = []
        data = []

        headers = copy.deepcopy(self.column_names)
        for a in other_table.column_names:
            try:
                key_column_names.index(a)
            except ValueError:
                headers.append(a)
        try:
            for name in key_column_names:
                table1.append(self.column_names.index(name))
                table2.append(other_table.column_names.index(name))
        except:
            pass
        else:
            for row1 in self.data:
                for row2 in other_table.data:
                    equal = True

                    for j in range(len(key_column_names)):
                        value1 = row1[table1[j]]
                        value2 = row2[table2[j]]
                        if value1 != value2:
                            equal = False
                            break
                    
                    if equal == True:
                        row = copy.deepcopy(row1)
                        for k in range(len(row2)):
                            try:
                                table2.index(k)
                            except ValueError:
                                row.append(row2[k])
                        
                        data.append(row)

        return MyPyTable(headers, data=data) 

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """
        
        table1 = []
        table2 = []
        header = []
        data = []

        header = copy.deepcopy(self.column_names)

        for a in other_table.column_names:
            try:
                key_column_names.index(a)
            except ValueError:
                header.append(a)
        try:
            for name in key_column_names:
                table1.append(self.column_names.index(name))
                table2.append(other_table.column_names.index(name))
        except:
            pass
        else:
            for row1 in self.data:
                added = False
                for row2 in other_table.data:
                    equal = True
                    for j in range(len(key_column_names)):
                        value1 = row1[table1[j]]
                        value2 = row2[table2[j]]
                        if value1 != value2:
                            equal = False
                            break
                    
                    if equal == True:
                        added = True
                        row = copy.deepcopy(row1)
                        for k in range(len(row2)):
                            try:
                                table2.index(k)
                            except ValueError:
                                row.append(row2[k])
                        
                        data.append(row)

                if added == False:
                    row = copy.deepcopy(row1)
                    nas = len(header) - len(self.column_names)
                    for k in range(nas):
                        row.append("NA")
                    
                    data.append(row)
            
            for row2 in other_table.data:
                match = False
                for row1 in data:
                    match = True
                    for j in range(len(key_column_names)):                       
                        value1 = row1[table1[j]]
                        value2 = row2[table2[j]]
                        if value1 != value2:
                            match = False
                            break
        
                    if match == True:
                        break

                if match == False:
                    row = copy.deepcopy(self.data[0])

                    for v in range(len(row)):
                        try:
                            i = table1.index(v)
                            row[v] = row2[table2[i]]
                        except ValueError:
                            row[v] = "NA"
                    
                    for i in range(len(row2)):
                        try:
                            table2.index(i)
                        except ValueError:

                            row.append(row2[i])
                    
                    data.append(row)

        return MyPyTable(header, data=data) 

    def get_frequencies(self, col_identifier):
        '''
        Returns values and their frequency from a column
        '''
        col = self.get_column(col_identifier)
        col.sort() # inplace 
        # parallel lists
        values = []
        counts = []
        for value in col:
            if value in values: # seen it before
                counts[-1] += 1 # okay because sorted
            else: # haven't seen it before
                values.append(value)
                counts.append(1)

        return values, counts
