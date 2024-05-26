# Access postgre database

import psycopg2
postgre_link = "postgresql://niphemi.oyewole:W7bHIgaN1ejh@ep-delicate-river-a5cq94ee-pooler.us-east-2.aws.neon.tech/Vetassist?statusColor=F8F8F8&env=&name=redditors%20db&tLSMode=0&usePrivateKey=false&safeModeLevel=0&advancedSafeModeLevel=0&driverVersion=0&lazyload=false"

from urllib.parse import urlparse
result = urlparse(postgre_link)
username = result.username
password = result.password
database = result.path[1:]
hostname = result.hostname
port = result.port
connection = psycopg2.connect(
    database = database,
    user = username,
    password = password,
    host = hostname,
    port = port
)

cursor = connection.cursor()

cursor.execute("SELECT * FROM reddit_usernames_comments;")
reddit_usernames_comments = cursor.fetchall()
cursor.execute("SELECT * FROM reddit_usernames;")
reddit_usernames = cursor.fetchall()
connection.commit()
connection.close()

#len(reddit_usernames_comments), len(reddit_usernames)

# Train: 3000, test: 1500, validation: 1500

TrainData = [red[:2] for red in reddit_usernames_comments[:3000]]
TestData= [red[:2] for red in reddit_usernames_comments[3000:4500]]
validationData = [red[:2] for red in reddit_usernames_comments[4500:]]