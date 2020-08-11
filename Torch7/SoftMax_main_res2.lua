require "torch"


opt={};
train={};
wkdir="/data/ted/WGD/SoftMax_"


dofile (wkdir.."Policy_res2.lua");

--Chrom_Model=torch.load(wkdir.."Model_Chrom_Model_res");
--CNV_Model=torch.load(wkdir.."Model_CNV_Model_res");
--End_Point_Model=torch.load(wkdir.."Model_End_Point_Model_res");


dofile (wkdir.."Advantage_res2.lua");
dofile (wkdir.."train_res2.lua");
dofile (wkdir.."data_simu_res2.lua");

cycle=100000000
counter=0000;
LoadData(true)

for c=0,cycle do
	counter=counter+1;
		flag=false;
		
	print(string.format("Start cycle %d", c));
    print("Loading data");
	LoadData(flag);
    print("Start train");
	model_train()
	Loss=torch.log(torch.sum(torch.pow(train.Advantage2,2))/train.Advantage:size(1))
    print(string.format("Loss: %6.6f",Loss));
	
	temp_state=train.state:clone()
	temp_next=train.next:clone()
	temp_valid=train.valid:clone()
	LoadData_chr()
	model_train()
	Error=torch.log(torch.sum(torch.pow(train.Advantage,2))/train.Advantage:size(1))
    print(string.format("Error: %6.6f",Error));
	train.state=temp_state
	train.next=temp_next
	train.valid=temp_valid
    print(string.format("step: %6.6f",train.step:sum()/train.step:size(1)))
    print("Save model");
 --   print(torch.sum(train.valid))
--	print(torch.sum(train.WGD_flag))
--	print(train.Advantage)
--	print(train.Reward)
--	print(Chrom_Model.output)
--	print(CNV_Model.output)
--	print(End_Point_Model.output)
    torch.save(wkdir.."Model_Chrom_Model_res",Chrom_Model);
    torch.save(wkdir.."Model_CNV_Model_res",CNV_Model);
	torch.save(wkdir.."Model_End_Point_Model_res",End_Point_Model);
--	torch.save(wkdir.."Model_Chrom_Model_old_com",Chrom_Model_old);


end
