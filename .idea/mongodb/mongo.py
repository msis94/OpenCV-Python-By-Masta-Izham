import pymongo
import datetime

#-------------------------------
date_time_str = str(datetime.datetime.now())
date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')

date = str(date_time_obj.date())
time = str(date_time_obj.time())
#-------------------------------

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

mydb = myclient["skymind"]
mycol_employee = mydb["employee"]
mycol_attendance = mydb["attendance"]

# To see how many database that exist in the local host
# print('Database Name : ', myclient.list_database_names())

# To check if the database name is exist or not
# dblist = myclient.list_database_names()
# if "skymind" in dblist:
#     print("The database exists.")

# To list the available collection in that database
# print('Collection Name :', mydb.list_collection_names())

# To see that collection are exist or not
# collist = mydb.list_collection_names()
# if "attendance" in collist:
#     print("The collection exists.")

# To insert one data in one time
# mydict_employee = {'emp_name': 'izham', 'emp_num': '083'}
# mycol_employee.insert_one({'emp_name': 'izham', 'emp_num': '083'})

# mydict_attendance = {'time_in': time, 'time_out': time}
mycol_attendance.insert_one({'date': date, 'emp_id':'083', 'time_in': time, 'time_out': time})

# x = mycol.insert_one(mydict)
# print(x.inserted_id)

# Find the fisrt data
# x = mycol.find_one()
# print(x)

# list all the data in collection
# for x in mycol.find():
#     print(x)

# list alll the data in collection, but only selected field
# for x in mycol.find({},{ "_id": 0, "date": 1, "time_in": 1 }):
#     print(x)

# To list all, except ids
# for x in mycol.find({},{ "_id": 0 }):
#     print(x)

#----------------------------------------
# MongoDB Query
#----------------------------------------

# myquery = { "emp_name": "izham" }
#
# mydoc = mycol.find(myquery)
#
# for x in mydoc:
#     print(x)

#------------------------
# update
#------------------------
#
myquery = { "emp_name": "izham" }
newvalues = { "$set": { "emp_name": "dollah" } }

mycol_employee.update_one(myquery, newvalues)

#print "customers" after the update:
for x in mycol_employee.find():
    print(x)