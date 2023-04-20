'''
batch converter for whole dataset from textgrid to txt
'''
'''
first cd sgh_TextGrid directory
'''

import textgrid
import glob
import argparse
import os

parser = argparse.ArgumentParser(description='arguments')

parser.add_argument('--data-path',type=str,default='./',help='path to TextGrids directory')

args = parser.parse_args()
path = os.path.join(args.data_path,'*.TextGrid')
#path = '/home/liuc0062/Malay_dataset/abax-klass-en-ml-cs-oct2022/*.TextGrid' #change to the folder with textgrid files
files = glob.glob(path)
count=0
direc = os.path.join(args.data_path,'../sgh_txt')
os.makedirs(direc, exist_ok=True)
for filename in files: 
    try:
        #print(filename)
        tg = textgrid.TextGrid.fromFile(filename)
        interval_num = len(tg[0])
        lst = []                                             
        for i in range(0,interval_num):
            mintime = tg[0][i].minTime
            maxtime = tg[0][i].maxTime
            mark = tg[0][i].mark
            string = filename.split('.')[0]+'.'+filename.split('.')[1]+'-{}-{} {} {} {}'.format(str(int(round(mintime*100,0))).zfill(6),
                                                                    str(int(round(maxtime*100,0))).zfill(6),
                                                                    mintime,
                                                                    maxtime,
                                                                    mark)
            lst.append(string)
        #print('done')  
        #print(direc,filename)
        if len(filename.split('.'))>2:
            with open(direc+filename.split('.')[1][13:]+'.txt', 'w') as f:
                f.write('\n'.join(lst))
        else:
            with open(os.path.join(direc,filename.split('.')[0]+'.txt'), 'w') as f:
                f.write('\n'.join(lst))
    except:
        #print(filename)
        continue

    # except:
    #         print(filename,i,"exception") # if there are only a few exceptions, might check those files whether there are some contents missing (e.g. intervals) or file formatting issue. 
    #         count+=1
    #         continue
#print(count)
