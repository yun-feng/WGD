##test the behavior of networks

require "torch"
require "nn"
require "nngraph"

wkdir="/data/ted/WGD/Max_"
opt={}
train={}

dofile (wkdir.."Policy.lua");
--dofile (wkdir.."Advantage.lua");
--dofile (wkdir.."train.lua");
--dofile (wkdir.."data.lua");

Chrom_Model=torch.load(wkdir.."Model_Chrom_Model");
CNV_Model=torch.load(wkdir.."Model_CNV_Model");
End_Point_Model=torch.load(wkdir.."Model_End_Point_Model");

dofile (wkdir.."Advantage2.lua");
dofile (wkdir.."train2.lua");
dofile (wkdir.."data_simu2.lua");
counter=20000

Chrom_Model.output

LoadData(1)

Chrom_Model:forward(train.state)

a,b=Chrom_Model.output:min(2)
x=torch.Tensor(25)
for i=1,25 do
x[i]=b[i]-train.ChrA[i]
end



train.next=torch.ones(1,2,22*50,1)
flag=false
LoadData(flag)

train.next=torch.ones(1,2,22*50,1)
for i =1,10 do
	train.next[1][1][i][1]=0
end
LoadData(flag)
