import pymongo
import datetime

#-------------------------------
# Configure the time
date_time_str = str(datetime.datetime.now())
date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')

date = str(date_time_obj.date())
time = str(date_time_obj.time())
#-------------------------------

#-------------------------------
# Setup mongo configuration
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["skymind"]
mycol_employee = mydb["employee"]
mycol_attendance = mydb["attendance"]
#-------------------------------
row_status = ''

# find the row, where emp_id='083' && date='2020-09-14'
myquery = { "emp_num": int('4'), "date":"2020-09-14" }

mydoc = mycol_attendance.find(myquery)

for row_status in mydoc:
    print(row_status)
    # print(row_status.get("time_out"))
    myquery = { "time_out": row_status.get("time_out") }
    newvalues = { "$set": { "time_out": time } }
    mycol_attendance.update_one(myquery, newvalues)

if(row_status==''):
    print('nothing')
    mycol_attendance.insert_one({'date': date, 'emp_num':int('4'), 'time_in': time, 'time_out': ''})