from random import shuffle

import numpy as np

import  os

SYNTHIA_PATH = os.environ['HOME'] + "/DL/Project/Dataset/SYNTHIA"

import cv2


class DB_SynthiaProvider:


    def __init__(self ,  batch_size):
        self.img_list = self.load_img_list();
        self.batch_size  = batch_size
        self.g_index = 0;
        self.size_img_list = len(self.img_list);
        shuffle(self.img_list)



    def load_img_list(self):
        return self.readFileLineByLine(SYNTHIA_PATH + "/ALL.txt");

    def readFileLineByLine(self, filename):
        all_lines = []
        with open( filename ) as f:
            all_lines = f.readlines()
        all_lines = [iLine.strip() for iLine in all_lines];
        return all_lines

    def load_single_instance(self, iImageName):
        img = self.load_img(SYNTHIA_PATH + "/RGB/" + iImageName);
        segmented_mask = self.load_img(SYNTHIA_PATH + "/GT/" + iImageName);
        iLabelName = iImageName.replace(".png" , ".txt");
        raw_labels = self.readFileLineByLine(SYNTHIA_PATH + "/GTTXT/" + iLabelName )

        segmented_labels = np.zeros(( 720 , 960) , dtype=np.int);
        for idx , iLine in enumerate(raw_labels):
            segmented_labels[idx] = iLine.split(' ');
        segmented_labels = segmented_labels[: , : , np.newaxis]
        return img , segmented_mask , segmented_labels


    def X_Synthia_Generator(self):
        ret_list = []
        l_list = []

        for iBatchCount in range(self.batch_size):
            iImageName = self.img_list[ self.g_index];
            img, segmented_mask, segmented_labels = self.load_single_instance(iImageName);
            ret_list.append( img  )
            l_list.append( segmented_labels)
            self.g_index = self.g_index + 1;
            if(self.g_index == self.size_img_list):
                break;


        yield np.asarray(ret_list) , np.asarray(l_list);


    def load_img(self , filename):
        img = cv2.imread( filename , cv2.IMREAD_UNCHANGED)
        return img


if __name__ == '__main__':
    batch_size = 10;

    d = DB_SynthiaProvider(batch_size);
    for iBatch in d.X_Synthia_Generator():

        img, segmented_labels = iBatch
        print("->");

        break;



