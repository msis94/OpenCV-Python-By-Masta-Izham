import pika

# establish the connection
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# the queue that we want to use, where we need to declare it's first
channel.queue_declare(queue='hello')

# There are 3 thing needs to be configured in order to send the message
# 1. Exchange
# 2. Routing key is the queue name that has been declared
# 3. Body
channel.basic_publish(exchange='',
                      routing_key='hello',
                      body='You are receiving this from Python!')

print(" [x] Message has been send")

# close the connection
connection.close()