require "torch"


opt={};

train.state=readdata()

dofile "/data/ted/cnv/Value.lua";
dofile "/data/ted/cnv/Policy.lua";


dofile "/data/ted/cnv/train.lua";
dofile "/data/ted/cnv/data.lua";

cycle=12000000;


for c = 1,cycle do
	if c%10==0 then
        print(string.format("Start cycle %d", c));
        print("Loading data");
		train.state=readdata();
		LoadData();
        print("Start train");
		training_loss=torch.Tensor(model_train());
        print(string.format("loss %6.6f",training_loss));
        print("Save model");
		torch.save("/data/ted/cnv/ValueNet",ValueNet);
        torch.save("/data/ted/cnv/ChromNet",ChromNet);
        torch.save("/data/ted/cnv/PolicyNet",UpperPolicyNet);
		torch.save("/data/ted/cnv/old_par",old_par);
	end
end