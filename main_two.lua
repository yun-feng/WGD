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
Loaddata(1,1);

for c=0,cycle do
	counter=counter+1;
	--remember baseline for first 1000 steps.
	if(counter<1000) then
		if torch.rand(1)[1]>0.03 then
                        flag=1;
                else
                        flag=0;
		end
		print(string.format("Start cycle %d", c));
                print("Loading data");
		LoadData(flag,1);
		print("Start train");
                Val_train();
                Reward_to_go=train.Reward:sum()/train.Reward:size(1)
                print(string.format("Average Reward to go %6.6f",Reward_to_go));
                print("Save model");
                torch.save(wkdir.."Model_ValueNet",ValueNet);
                torch.save(wkdir.."Model_ValueNet_eval",ValueNet_eval);
	elseif(torch.ceil(counter/1000)%2==1) then
		if torch.rand(1)[1]>0.03 then
                        flag=1;
                else
                        flag=0;
                end
		print(string.format("Start cycle %d", c));
                print("Loading data");
                LoadData(flag,false);
		print("Start train");
		Policy_train();
                Reward_to_go=train.Reward:sum()/train.Reward:size(1)
                print(string.format("Average Reward to go %6.6f",Reward_to_go));
                print("Save model");
                torch.save(wkdir.."Model_Chrom_Model",Chrom_Model);
                torch.save(wkdir.."Model_CNV_Model",CNV_Model);
                torch.save(wkdir.."Model_End_Point_Model",End_Point_Model);
	else
		if torch.rand(1)[1]>0.03 then
                        flag=1;
                else
                        flag=0;
                end
                print(string.format("Start cycle %d", c));
                print("Loading data");
                LoadData(flag,false);
                print("Start train");
                Val_train();
                Reward_to_go=train.Reward:sum()/train.Reward:size(1)
                print(string.format("Average Reward to go %6.6f",Reward_to_go));
                print("Save model");
                torch.save(wkdir.."Model_ValueNet",ValueNet);
                torch.save(wkdir.."Model_ValueNet_eval",ValueNet_eval);
	end
end
