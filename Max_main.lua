require "torch"


opt={};
train={};
wkdir="/data/ted/WGD/Max_"


dofile (wkdir.."Policy.lua");

dofile (wkdir.."Advantage2.lua");
dofile (wkdir.."train2.lua");
dofile (wkdir.."data_simu2.lua");

cycle=100000000
counter=0;
LoadData(true)

for c=0,cycle do
	counter=counter+1;
--	if torch.rand(1)[1]>0.99 then
--		flag=true;
--	else 
		flag=false;
--	end
		
	print(string.format("Start cycle %d", c));
    print("Loading data");
	LoadData(flag);
--Advantage_cal();
    print("Start train");
	model_train();
	Error=torch.log(torch.sum(torch.pow(train.Advantage,2))/train.Advantage:size(1))
    print(string.format("Error: %6.6f",Error));
	Loss=torch.log(torch.sum(torch.pow(train.Advantage2,2))/train.Advantage:size(1))
    print(string.format("Loss: %6.6f",Loss));
    temp=train.state:clone()
    LoadData_Reverse()
    model_train()
    train.state=temp
    print("Save model");
    torch.save(wkdir.."Model_Chrom_Model",Chrom_Model);
    torch.save(wkdir.."Model_CNV_Model",CNV_Model);
	torch.save(wkdir.."Model_End_Point_Model",End_Point_Model);
	

end
