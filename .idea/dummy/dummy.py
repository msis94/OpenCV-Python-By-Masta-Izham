import pymongo

#-------------------------------
# Setup mongo configuration
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["skymind"]
mycol_employee = mydb["employee"]
mycol_attendance = mydb["attendance"]
#-------------------------------

emp_num = 1

myquery = { "emp_num": emp_num}
mydoc = mycol_employee.find(myquery)

emp_nume = 0
for row_status in mydoc:
    emp_nume = row_status.get("emp_num")
    # print(row_status.get("emp_name"))

print(emp_nume)