
#rm -r mongodb/dbfolder
#rm    mongodb/log/log1.txt
mkdir mongodb/dbfolder82

#sleep 1

mongod --fork --logpath mongodb/log/log_82.txt --dbpath mongodb/dbfolder82/
python2 Spearmint/spearmint/main.py ./