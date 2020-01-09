rm -r mongodb/ganfolder
mkdir mongodb/ganfolder
rm    mongodb/log/gan.txt
#mongod --fork --logpath mongodb/log/gan.txt --dbpath mongodb/ganfolder/
mongod --logpath mongodb/log/gan.txt --dbpath mongodb/ganfolder/

python2 Spearmint/spearmint/main.py --config gan_spear_config.json ./
