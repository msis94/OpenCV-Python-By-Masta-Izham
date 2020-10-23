import pandas as pd

# asking user to enter employee number
emp_num =int(input("Please enter Employee Number : "))

# open databese file
df = pd.read_csv("/home/goblin/Desktop/face_database.csv")

print(df, type(df))



print('-----------------------------------------------')

# find the row of employee number (eg. = 7)
emp_data = df.loc[df["employee number"]==emp_num]
print(emp_data, type(emp_data))

print('-----------------------------------------------')

# after found employee number show the name of the employee by taking
# the -name- column
emp_data = emp_data["name"].values[0]
print(emp_data, type(emp_data))


print("Inilah :", emp_data)
