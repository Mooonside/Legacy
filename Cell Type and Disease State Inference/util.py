import numpy as np
import os

def division(array,labels,ratio=0.2):
    '''
    divide the dataset according to their type
    ratio means the ratio for test data
    '''
    uniq = np.unique(labels)

    for i in uniq:
        mask = (labels == i)
        mask = np.nonzero(mask)[0]
        mask_len = len(mask)
        div = int(ratio * mask_len)
        random = np.random.choice(mask_len,mask_len,replace=False)
        train_mask = mask[0:mask_len - div]
        test_mask = mask[mask_len - div:mask_len]
        if(i==uniq[0]):
            train_data = array[train_mask,:]
            test_data = array[test_mask,:]
            train_label = labels[train_mask]
            test_label = labels[test_mask]
        else:
            train_data = np.vstack((train_data,array[train_mask,:]))
            train_label = np.vstack((train_label,labels[train_mask]))
            test_data = np.vstack((test_data,array[test_mask,:]))
            test_label = np.vstack((test_label,labels[test_mask]))
    return train_data,train_label,test_data,test_label


def read_data(mode = "CellType",th = 5):
    # read in microarray info , arrayid is stored in idlist

    keymap = {"CellType":4,"DiseaseStage":7}
    mapidx = keymap[mode]

    TypeCnt = {}
    
    arrayname = "./res/%s_Array_th_%d.npy" % (mode,th)
    labelname = "./res/%s_Label_th_%d.npy" % (mode,th)
    
    if (os.path.exists(arrayname) and os.path.exists(labelname)):
        print "Loading Previous Data!"
        array = np.load(arrayname)
        label = np.load(labelname)
        return array,label
     
    i = 0
    with open("Gene_Chip_Data/E-TABM-185.sdrf.txt") as f:
        for line in f:
            splitlist = line.split("\t")
            if(i == 0):
                pass
            else:
                value = splitlist[mapidx]
                # throw away datas with no labels
                if(value != "  "):
                    if(value in TypeCnt.keys()):
                        TypeCnt[value] += 1
                    else:
                        TypeCnt[value] = 1
            i += 1


    TypeIdx = {}
    idx = 0
    for i in sorted(TypeCnt.keys()):
        # filter those less than th labels
        if(TypeCnt[i] >= th):
            TypeIdx[i] = idx
            idx += 1

    i = 0
    RawLabels = []
    with open("Gene_Chip_Data/E-TABM-185.sdrf.txt") as f:
        for line in f:
            splitlist = line.split("\t")
            if(i == 0):
                pass
            else:
                value = splitlist[mapidx]
                if(value in TypeIdx.keys()):
                    RawLabels.append(TypeIdx[value])
                else:
                    RawLabels.append(-1)
            i += 1
            
    # read in microarray info , arrayid is stored in idlist
    if os.path.exists("./res/raw_array.npy"):
        RawArray = np.load("./res/raw_array.npy")
    else:
        # GeneID = []
        # ProbeSetID = []
        RawArray = np.zeros([5896,22283])

        i = 0
        with open("Gene_Chip_Data/microarray.original.txt") as f:
            for line in f:
                splitlist = line.split("\t")
                if(i == 0):
                    pass
                    # for j in splitlist:
                    #     ProbeSetID.append(j)
                else:
                    # GeneID.append(splitlist[0])
                    RawArray[:,i-1] = np.asarray(splitlist[1:])
                i += 1
        np.save("./res/raw_array.npy",RawArray)
    
    Labels = np.asarray(RawLabels)
    mask = (Labels != -1)

    MicroArray = RawArray[mask,:]
    Labels = Labels[mask]
    Labels = Labels.reshape([-1,1])

    np.save(arrayname,MicroArray)
    np.save(labelname,Labels)

    print "Data Generated And Stored!"  
    return MicroArray,Labels

def pca(array,ratio=0.95):
    epsilon = 1e-5
    mean = np.mean(array,axis=0)
    #var = np.var(array,axis=0)
    NormArray = array - mean
    U, sing_vals, V = np.linalg.svd(NormArray, full_matrices=False)
    sing_cumsum = np.cumsum(sing_vals)
    sing_ratio = sing_cumsum/np.sum(sing_vals)
    cut = len(sing_ratio)-1
    for i in range(len(sing_ratio)-1,0,-1):
        if(sing_ratio[i] < ratio + epsilon):
            cut = i
            break
    print cut
    U_cut = U[:,0:cut]
    sing_cut = sing_vals[0:cut]
    ProjArray = U_cut.dot(np.diag(sing_cut))
    return ProjArray
    
