import datetime
import pymongo
import time

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

mydb = myclient["Confab"]
mycol_attendance = mydb["Attendance"]

# # python change date to milisecond
#
# # change date to number
date = int(datetime.datetime.now().timestamp() * 1000)
print("Date : ",int(date))
# # change back to datetime
print(datetime.datetime.fromtimestamp(1601953200000/1000))

# mycol_attendance.insert_one({'date': date, 'emp_id':'083'})

