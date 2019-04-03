require "torch"
require "nn"
require "nngraph"

wkdir="/data/ted/WGD/Max_"
opt={}
train={}
test={}

dofile (wkdir.."Policy2.lua");
batch_sample=10
--dofile (wkdir.."Advantage.lua");
--dofile (wkdir.."train.lua");
--dofile (wkdir.."data.lua");

Chrom_Model=torch.load(wkdir.."Model_Chrom_Model_com");
CNV_Model=torch.load(wkdir.."Model_CNV_Model_com");
End_Point_Model=torch.load(wkdir.."Model_End_Point_Model_com");

dofile (wkdir.."Advantage2.lua");
dofile (wkdir.."train2.lua");
dofile (wkdir.."data_simu2.lua");

train.step_sample=10

train.Chr_sample=torch.zeros(batch_sample,train.step_sample)
train.CNV_sample=torch.zeros(batch_sample,train.step_sample)
train.End_sample=torch.zeros(batch_sample,train.step_sample)
train.WGD_sample=torch.zeros(batch_sample)

LoadData_t(train.step_sample)

test.ChrA=torch.zeros(train.step_sample+10)
test.CNV=torch.zeros(train.step_sample+10)
test.End=torch.zeros(train.step_sample+10)

Deconvolute(train.state[2],train.step_sample+10)

torch.sum(torch.abs(train.state[2]-1))
