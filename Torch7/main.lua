require "torch"


opt={};
train={};
wkdir="/data/ted/WGD/"


dofile (wkdir.."Value.lua");
dofile (wkdir.."Policy.lua");

ValueNet=torch.load(wkdir.."Model_ValueNet_eval2")
ValueNet_eval=ValueNet:clone()

dofile (wkdir.."Advantage.lua");
dofile (wkdir.."train.lua");
dofile (wkdir.."data.lua");

cycle=100000000
counter=0;
for c=0,cycle do
	counter=counter+1;
	if c%10==0 then
		if torch.rand(1)[1]>0.5 then
			flag=1;
		else 
			flag=0;
		end
		if(counter<1000) then
			flag2=1;
			opt.State_Val_eval.learnningRate=1;
		else
			flag2=false;
			opt.State_Val_eval.learningRate=0.01;
        	end
		if torch.rand(1)[1]<0.2 then
			flag2=1;
		end	
		print(string.format("Start cycle %d", c));
        	print("Loading data");
		LoadData(flag,flag2);
	        print("Start train");
		if counter<1000 then
			Val_train();
		else
			model_train();
		end
		Reward_to_go=train.Reward:sum()/train.Reward:size(1)
        	print(string.format("Average Reward to go %6.6f",Reward_to_go));
        	print("Save model");
		torch.save(wkdir.."Model_ValueNet",ValueNet);
		torch.save(wkdir.."Model_ValueNet_eval",ValueNet_eval);
        	torch.save(wkdir.."Model_Chrom_Model",Chrom_Model);
        	torch.save(wkdir.."Model_CNV_Model",CNV_Model);
		torch.save(wkdir.."Model_End_Point_Model",End_Point_Model);
	else
		flag=false;
		flag2=false;
		LoadData(flag,flag2);
		Val_train();
		torch.save(wkdir.."Model_ValueNet",ValueNet);
		torch.save(wkdir.."Model_ValueNet_eval",ValueNet_eval);
        --	torch.save(wkdir.."Model_Chrom_Model",Chrom_Model);
        --	torch.save(wkdir.."Model_CNV_Model",CNV_Model);
	--	torch.save(wkdir.."Model_End_Point_Model",End_Point_Model);
	end
	if c%100==0 then
		flag=false;
		flag2=1;
		train.next=torch.ones(100,2,22*50,1)
		LoadData(flag,flag2);
		model_train();
	end
end
