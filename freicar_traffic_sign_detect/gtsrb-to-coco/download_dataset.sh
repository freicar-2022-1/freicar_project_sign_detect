#!/bin/bash

wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
unzip GTSRB_Final_Training_Images.zip
rm GTSRB_Final_Training_Images.zip

wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
unzip GTSRB_Final_Test_Images.zip
rm GTSRB_Final_Test_Images.zip

wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip
unzip GTSRB_Final_Test_GT.zip
rm GTSRB_Final_Test_GT.zip
mv GT-final_test.csv GTSRB/Final_Test/Images/