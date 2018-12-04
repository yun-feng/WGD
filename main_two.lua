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
LoadData(1,1);
flag_up=false;

for c=0,cycle do
	counter=counter+1;
	--remember baseline for first 1000 steps.
	if(counter<1000) then
		opt.State_Val_eval.learningRate=1;
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
	elseif(torch.floor(counter/1000)%3==1) then
		if not flag_up then
			flag_up=1;
			print("Update Eval Net")
			ValueNet_eval=ValueNet:clone()
		end
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
                print("Save policy model");
                torch.save(wkdir.."Model_Chrom_Model",Chrom_Model);
                torch.save(wkdir.."Model_CNV_Model",CNV_Model);
                torch.save(wkdir.."Model_End_Point_Model",End_Point_Model);
	else
		opt.State_Val_eval.learningRate=0.01
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
                print("Save value model");
                torch.save(wkdir.."Model_ValueNet",ValueNet);
                torch.save(wkdir.."Model_ValueNet_eval",ValueNet_eval);
		flag_up=false;
	end
end
