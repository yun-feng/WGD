require "torch"


opt={};
train={};
wkdir="/data/ted/WGD/Max_"


dofile (wkdir.."Policy_res.lua");

--Chrom_Model=torch.load(wkdir.."Model_Chrom_Model_res");
CNV_Model=torch.load(wkdir.."Model_CNV_Model_com");
End_Point_Model=torch.load(wkdir.."Model_End_Point_Model_com");


dofile (wkdir.."Advantage_res.lua");
dofile (wkdir.."train_res.lua");
dofile (wkdir.."data_simu_res.lua");

cycle=100000000
counter=0000;
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
--train.Advantage2:zero()
--train.Advantage:zero()
--Advantage_cal();
    print("Start train");
--	split_train=1;
--	temp_Advantage2=train.Advantage2:clone();
--	temp_Advantage=train.Advantage:clone()
--	while(torch.log(torch.sum(torch.pow(train.Advantage2,2))/train.Advantage:size(1)) > 6) do
--		split_train=split_train+1;
--		train.Advantage2=temp_Advantage2/split_train
--	end
--	temp_split=split_train
--	while (split_train>0.5) do
--		model_train();
--		Advantage_cal()
--		train.Advantage2=train.Advantage2/temp_split
--		train.Advantage=train.Advantage/temp_split
--		split_train=split_train-1;
--	end
	model_train()
	Error=torch.log(torch.sum(torch.pow(train.Advantage,2))/train.Advantage:size(1))
    print(string.format("Error: %6.6f",Error));
	Loss=torch.log(torch.sum(torch.pow(train.Advantage2,2))/train.Advantage:size(1))
    print(string.format("Loss: %6.6f",Loss));
    print(string.format("step: %6.6f",train.step:sum()/train.step:size(1)))
  --  if torch.rand(1)[1]> 0.8/(1+2*math.exp(-2e-4*counter)) then
--	temp=train.state:clone()
--	temp_next=train.next:clone()
--	temp_valid=train.valid:clone()
--	temp_Advantage=train.Advantage2:clone()
--	LoadData_Reverse()
--	        split_train=0;
  --      while(torch.log(torch.sum(torch.pow(train.Advantage2,2))/train.Advantage:size(1)) > 6) do
    --            split_train=split_train+1;
      --          train.Advantage2=train.Advantage2/2
	--	train.Advantage=train.Advantage/2
--        end
  --      while (split_train>0.5) do
    --            model_train();
      --          split_train=split_train-1;
        --end
--	model_train()
--	train.state=temp
--	train.valid=temp_valid
--	train.Advantage2=temp_Advantage2
--	train.next=temp_next
  --  end
    print("Save model");
    torch.save(wkdir.."Model_Chrom_Model_res",Chrom_Model);
    torch.save(wkdir.."Model_CNV_Model_res",CNV_Model);
	torch.save(wkdir.."Model_End_Point_Model_res",End_Point_Model);
--	torch.save(wkdir.."Model_Chrom_Model_old_com",Chrom_Model_old);


end
