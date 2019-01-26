require "torch"


opt={};
train={};
wkdir="/data/ted/WGD/Softmax_"


dofile (wkdir.."Policy.lua");

dofile (wkdir.."Advantage.lua");
dofile (wkdir.."train.lua");
dofile (wkdir.."data.lua");

cycle=100000000
counter=0;
LoadData(true)

for c=0,cycle do
	counter=counter+1;
	if torch.rand(1)[1]>0.99 then
		flag=true;
	else 
		flag=false;
	end
		
	print(string.format("Start cycle %d", c));
    print("Loading data");
	LoadData(flag);
    print("Start train");
	model_train();
	Error=torch.log(torch.sum(torch.pow(train.Advantage,2))/train.Advantage:size(1))
    print(string.format("Error: %6.6f",Error));
    print("Save model");
    torch.save(wkdir.."Model_Chrom_Model",Chrom_Model);
    torch.save(wkdir.."Model_CNV_Model",CNV_Model);
	torch.save(wkdir.."Model_End_Point_Model",End_Point_Model);
end
