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

# to find existing time out
for old_time in mycol_attendance.find({'time_out'}):
    print(old_time)


required_date=''

# if date is exist --> just find the row and update the time_out
for required_date in mycol_attendance.find({ "date": date }):
    print('exist')
    # update the(time out)
    myquery = { "time_out": existing }
    newvalues = { "$set": { "time_out": time } }
    mycol_attendance.update_one(myquery, newvalues)

# if date is not exist --> insert new data (fresh)
if(required_date==''):
    print('nothing')
    # insert(date, emp_id, time_in, time_out='')
    mycol_attendance.insert_one({'date': date, 'emp_id':'083', 'time_in': time, 'time_out': time})