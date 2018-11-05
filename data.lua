require "torch"

require "math"

chrom_extract=function(cnp,chrom)
	return cnp[{{50*(chrom-1)+1,50*chrom},{}}];
end


Rewarding=function(ChrA,FocA,StartL,EndL,StartS,EndS)
	local reward;
	if ChrA==1 then
		reward=torch.sum(torch.abs(StartS-1))*log(c/b);
		reward=reward-ValueNet:forward(EndS);
	else 
		if chrA==2 then
			reward=torch.sum(torch.abs(StartS-2*EndS))*log(b/c);
			reward=reward+log(WGD);
		else:
			reward=log(c/(b+log(EndL-StartL)));
			if torch.abs(EndL-StartL-25)<2 then
				reward=log(HC);
			else 
				if torch.abs(EndL-StartL-50)<3 then
					reward=log(WC);
				end
			end
		end
	end
	return reward;
end


	b=1
	c=0.1
	WGD=0.2
	WC=0.04
	HC=0.05

LoadData=function()
	local temp
	temp,train.ChromAct=torch.max(ChromNet:forward(train.state),2)
	train.focus=torch.Tensor(train.state:size(1),50)
	train.focus:fill(1);
	for i=1,train.state:size(1) do
		if(train.ChromAct[i]>2) then
			train.focus[i]=chrom_extract(train.state[i],train.ChromAct[i]-2)
		end
	end
	temp,train.FocusAct=torch.max(UpperPolicyNet:forward({train.state,train.focus}),2)
	train.FocusAct=train.FocusAct+2
	train.start=(train.ChromAct-3)*50+(train.FocusAct-2)%5+1
	for i=1,train.state:size(1) do
		train.end[i]=(train.ChromAct-2)*50
		while (torch.all(torch.ne(train.state[train.end[i]],train.state[train.start[i]]))) do
			train.end[i]=train.end[i]-1
		end
	end
	CNV_action=torch.Tensor({{1,0},{-1,0},{1,-1},{0,1},{0-1}})
	for i=1,train.state:size(1) do
		if train.ChromAct[i]==1 then
			train.next[i]=train.state[i]
		else 
			if train.ChromAct[i]==2 then
				train.next[i]=(train.state[i]/2):int()
			else 
				train.next[i]=(train.state[i]):clone()
				train.next[i][train.start[i]:train.end[i]]=train.next[i][train.start[i]:train.end[i]]+CNV_action[train.FocusAct%5]
			end
		end
	end
	for i=1,train.state:size(1) do
		train.reward[i]=Rewarding(train.ChromAct[i],train.FocusAct[i],train.start[i],train.end[i],train.state[i],train.next[i])
	end
end