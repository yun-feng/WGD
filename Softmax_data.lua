require "torch"

require "math"

cnp_file=("/data/ted/WGD/Simulation_cnp.txt")

chrom_extract=function(cnp,chrom)
	if chrom<=2 then
		return torch.zeros(nfeats,chrom_width,1);
	else
		return cnp[{{},{chrom_width*(chrom-3)+1,chrom_width*(chrom-2)},{}}];
	end
end

CNV_action=torch.Tensor({{1,0},{-1,0},{0,1},{0,-1}})

LoadData=function(flag)
	--when flag is true, load data from file
	--otherwise use the next state as input
	cnp={}
	cnpfile = io.open(cnp_file,"r")
	if cnpfile then
		local counter=0;
		local x=torch.floor(torch.rand(1)*10)[1]
		for cnpline in cnpfile:lines() do
			if((math.floor(counter/2))%10==x) then
				table.insert(cnp,cnpline:split("\t"));
			end
			counter=counter+1;
		end
        end
	train.state=torch.Tensor(cnp);
	cnp=nil;
	train.state:resize(train.state:size(1)/2,2,train.state:size(2),1);

	if not flag then
		for i=1,train.state:size(1) do
				train.state[i]=train.next[i];
		end
	end
	
	
	local temp
	--temp,train.ChrA=torch.max(ChromNet:forward(train.state),2)
	--sample chromosome
	--Extract the cnp for the chromosome
	
	train.ChrA=torch.floor(torch.rand(train.state:size(1))*(22+1))+2;
	train.chrom_state=torch.Tensor(train.state:size(1),nfeats,chrom_width,1);
	for i=1,train.ChrA:size(1) do
		train.chrom_state[i]=chrom_extract(train.state[i],train.ChrA[i])
	end
	
	--sample cnv
	--prepare potential cnp
	
	train.CNV=torch.floor(torch.rand(train.state:size(1))*(4*chrom_width))+1;
	train.StartL=torch.zeros(train.state:size(1));
	train.chrom_state_new=torch.Tensor(train.state:size(1),nfeats,chrom_width,1);
	for i=1,train.CNV:size(1) do
		train.chrom_state_new[i]=train.chrom_state[i];
		if train.ChrA[i]>2 then
			train.StartL[i]=math.ceil(train.CNV[i]/CNV_action:size(1));
			cnv_a=train.CNV[i]%CNV_action:size(1)+1;
			for j=train.StartL[i],chrom_width do
				train.chrom_state_new[i][1][j][1]=train.chrom_state_new[i][1][j][1]+CNV_action[cnv_a][1]
				train.chrom_state_new[i][2][j][1]=train.chrom_state_new[i][2][j][1]+CNV_action[cnv_a][2]
			end
		end
	end
	
	--sample end point
	
	train.End=torch.zeros(train.state:size(1));
	for i=1,train.End:size(1)do
		if train.ChrA[i]>2 then
			train.End[i]=train.StartL[i]+torch.floor(torch.rand(1)[1]*(chrom_width-train.StartL[i]+1));
		end
	end
	
	--Compute CNP at the next time point
	--Compute Reward-to-go and Advantage
	train.next=train.state:clone();
	train.Reward=torch.Tensor(train.state:size(1));
	local startL,endL;
	for i=1,train.Reward:size(1) do
		if train.ChrA[i]==2 then
			train.next[i]=(train.next[i]/2):floor()
		elseif train.ChrA[i]>2 then
			startL=chrom_width*(train.ChrA[i]-3)+train.StartL[i];
			endL=chrom_width*(train.ChrA[i]-3)+train.End[i];
			cnv_a=train.CNV[i]%CNV_action:size(1)+1;
			for j=startL,endL do
				train.next[i][1][j][1]=math.max(0,train.next[i][1][j][1]+CNV_action[cnv_a][1]);
				train.next[i][2][j][1]=math.max(0,train.next[i][2][j][1]+CNV_action[cnv_a][2]);
			end
		end
		train.Reward[i]=Reward(train.ChrA[i],startL,endL,train.state[i],train.next[i])
	end	
	train.Advantage=torch.Tensor(train.state:size(1));
	Advantage_cal();
end
