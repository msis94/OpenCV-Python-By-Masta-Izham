import pymongo


# connecting to client
def client(connection):
    try:
        myclient = pymongo.MongoClient(connection)
        print("Successfully connected to the client")
        return myclient

    except Exception as e:
        print("Connection Error : {}".format(e))

# connecting to client
client = client("mongodb://localhost:27017/")

# connecting to database

# connecting to collection
# dropping collection
# limit



# insert
# find
# query
# sort
# delete