from itertools import chain, zip_longest

from indian_road_gen import DB_IndiaProvider
from synthia_gen import DB_SynthiaProvider

import numpy as np

class GBL:
    
    def __init__(self , batch_size):

        self.india = DB_IndiaProvider(batch_size);
        self.synthia = DB_SynthiaProvider(batch_size);
        

    def global_db_gen(self  ):

        #zip_longest(self.india.X_India_Rd_Generator(), self.synthia.X_Synthia_Generator())


        for _ in range( int(self.india.size_img_list/self.synthia.batch_size) ):

            ret_list = []
            for iBatchCount in range(self.synthia.batch_size):
                iImageName = self.synthia.img_list[self.synthia.g_index];
                source_img, _, a = self.synthia.load_single_instance(iImageName);
                import cv2

                source_img = cv2.resize(source_img, (240, 180) , interpolation = cv2.INTER_NEAREST )
                #a = np.squeeze( a , axis=2)
                a = cv2.resize(a, (240, 180) ,  	interpolation = cv2.INTER_NEAREST )

                segmented_labels = (np.arange( 11 + 1) == a[..., None] - 1).astype(int)
                #segmented_labels = segmented_labels.squeeze(axis=(2,))

                #iImageName = self.india.img_list[self.india.g_index];
                #iImageName = None

                indImgName = self.india.img_list[self.india.g_index];
                target_img = self.india.load_single_instance(indImgName);

                target_img = cv2.resize(target_img, (240, 180), interpolation=cv2.INTER_NEAREST)


                ret_list.append( ( source_img  ,target_img , segmented_labels) )


                self.synthia.g_index = self.synthia.g_index + 1;
                self.india.g_index = self.india.g_index + 1;

                if (self.india.g_index == self.india.size_img_list):
                    print("Breaking")
                    break;

                # if (self.india.g_index == self.india.size_img_list):
                #     self.india.g_index = 0;


            yield ret_list





if __name__ == '__main__':
    batch_size =20;

    gbl = GBL(batch_size)
    index = 0
    for iCombined in gbl.global_db_gen():
        source_img, target_img ,segmented_labels  = iCombined[1];


        #name , img = iIndia[0]
        print("->" , index)# , target_img.shape  , source_img.shape , segmented_labels.shape)
        index = index + 1
	
