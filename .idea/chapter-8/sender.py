import pika
import json

emp_num = 3
date = '2/5/2020'
time_in = '3'
time_out = '4'

# a = "{\"emp_num\": "+str(emp_num)+", \"date\": \""+date+"\", \"time_in\": \""+time_in+"\", \"time_out\": \""+time_out+"\"}"
# print(a)

# a Python object (dict):
x = {
        "emp_num": emp_num,
        "date": date,
        "time_in": time_in,
        "time_out": time_out
    }

# convert into JSON:
y = json.dumps(x)

print(y)


# ---------------------------------------------------------------------

# establish the connection
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
attendance = connection.channel()

# the queue that we want to use, where we need to declare it's first
attendance.queue_declare(queue='attendance')

# There are 3 thing needs to be configured in order to send the message
# 1. Exchange
# 2. Routing key is the queue name that has been declared
# 3. Body

attendance.basic_publish(exchange='',
                         routing_key='attendance',
                         body=y)

print(" [x] Message has been send")

# close the connection
connection.close()