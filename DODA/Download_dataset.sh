
# Download GWHD
GWHD_Dir=datasets
mkdir -p $GWHD_Dir

wget https://zenodo.org/records/5092309/files/gwhd_2021.zip -O $GWHD_Dir/gwhd_2021.zip
unzip $GWHD_Dir/gwhd_2021.zip -d $GWHD_Dir