import mysql.connector

print("MySQl Community Sever Connector version is : {}".format(mysql.connector.__version__))

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="123"
)

print(mydb)