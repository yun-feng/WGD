##test the behavior of networks

require "torch"
require "nn"
require "nngraph"

wkdir="/data/ted/WGD/"
opt={}
train={}

dofile (wkdir.."Value.lua");
dofile (wkdir.."Policy.lua");
dofile (wkdir.."Advantage.lua");
dofile (wkdir.."train.lua");
dofile (wkdir.."data.lua");

ValueNet=torch.load(wkdir.."Model_ValueNet");
ValueNet_eval=torch.load(wkdir.."Model_ValueNet_eval");
Chrom_Model=torch.load(wkdir.."Model_Chrom_Model");
CNV_Model=torch.load(wkdir.."Model_CNV_Model");
End_Point_Model=torch.load(wkdir.."Model_End_Point_Model");

train.next=torch.ones(1,2,22*50,1)
flag=false
LoadData(flag)