import glob
from random import shuffle

import numpy as np
import os

INDIA_RD_PATH = os.environ['HOME'] + "/DL/Project/Dataset/"

import cv2


class DB_IndiaProvider:


    def __init__(self , batch_size):
        self.img_list = self.load_img_list();
        self.batch_size = batch_size
        self.g_index = 0;
        self.size_img_list = len(self.img_list);
        from random import shuffle
        shuffle(self.img_list)

    def load_img_list(self):
        return glob.glob( INDIA_RD_PATH + "out*.png" )

    def readFileLineByLine(self, filename):
        all_lines = []
        with open( filename ) as f:
            all_lines = f.readlines()
        all_lines = [iLine.strip() for iLine in all_lines];
        return all_lines

    def load_single_instance(self, iImageName):
        img = self.load_img( iImageName);
        return img

    def X_India_Rd_Generator(self):
        ret_list = []
        for iBatchCount in range(self.batch_size):
            iImageName = self.img_list[self.g_index];
            img  = self.load_single_instance(iImageName);
            ret_list.append( img)

            self.g_index = self.g_index + 1;
            if (self.g_index == self.size_img_list):
                break;
        yield np.asarray(ret_list);



    def load_img(self , filename):
        img = cv2.imread( filename , cv2.IMREAD_UNCHANGED)
        return img


if __name__ == '__main__':
    batch_size = 10;


    d = DB_IndiaProvider(batch_size);

    for iBatch in d.X_India_Rd_Generator():

        name , img = iBatch[0]
        #print(img);
        break;


