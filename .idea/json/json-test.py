import json

emp_num = 3
date = '2/5/2020'
time_in = '3'
time_out = '4'


# a Python object (dict):
x = {
    "emp_num": emp_num,
    "date": date,
    "time_in": time_in,
    "time_out": time_out
    }

# convert into JSON:
y = json.dumps(x)

# the result is a JSON string:
print(y)

json_object = json.loads(y)

print(json_object["time_out"])