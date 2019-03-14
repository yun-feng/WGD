require "torch"

require "math"


chrom_extract=function(cnp,chrom,allele)
	--if chrom<=1 then
	--	return torch.zeros(1,chrom_width,1);
	--else
		return cnp[{{allele,allele},{chrom_width*(chrom-1-allele*22+22)+1,chrom_width*(chrom-allele*22+22)},{}}];
--	end
end

CNV_action=torch.Tensor({{1,0},{-1,0},{0,1},{0,-1}})

LoadData=function(flag)
	--when flag is true, load data from file
	--otherwise use the next state as input
	if flag then
		train.state=torch.ones(25,2,1100,1)
        	train.next=torch.ones(25,2,1100,1)
		train.Advantage=torch.zeros(25)
		train.step=torch.zeros(25)-1
		train.valid=torch.ones(25)
		train.Advantage2=torch.zeros(25)
		train.WGD=torch.zeros(25)
	end

--	if not flag then
--		train.next=train.state:clone()
--	end
		
	train.ChrA=torch.floor(torch.rand(train.state:size(1))*(22*2))+1;
	train.StartL=torch.floor(torch.rand(train.state:size(1))*chrom_width)+1
	train.End=torch.floor(torch.cmul(torch.rand(train.state:size(1)),(chrom_width-train.StartL+1)))+train.StartL
	--train.allele=torch.floor(torch.rand(train.state:size(1))*2)+1
	for i=1,train.ChrA:size(1) do
		if(torch.rand(1)[1]>0.98/(1+math.exp(-2e-4*counter+2))  or torch.abs(train.Advantage2[i])>=5) then
			train.state[i]=torch.ones(2,1100,1)
			train.next[i]=torch.ones(2,1100,1)
			train.Advantage[i]=0
			train.step[i]=0
			train.WGD[i]=0
		else
			if((train.valid[i]>0) and ( torch.abs(train.Advantage2[i])<3)) then
				train.next[i]=train.state[i]:clone()
				train.step[i]=train.step[i]+1
				train.Advantage[i]=0--train.Advantage[i]*(1/train.step[i])
			else
				train.state[i]=train.next[i]:clone()
				train.Advantage[i]=0
			end
	
		end
		if(torch.rand(1)[1]>0.7) then
			train.StartL[i]=1
			train.End[i]=chrom_width
		end
	--	if((not flag) and torch.rand(1)[1]>0.7) then
	--		temp,train.ChrA[i]=Chrom_Model.output[torch.floor(torch.rand(1)[1]*train.state:size(1))+1]:min(1)
	--	end
	end
	
	train.allele=torch.floor((train.ChrA-1)/22)+1
	--train.cnv=(torch.floor(torch.rand(train.state:size(1))*2))
	train.cnv=torch.zeros(train.state:size(1))
	for i=1,train.ChrA:size(1) do
		if(torch.rand(1)[1]>0.7) then
			train.cnv[i]=1
		end
	end
	train.CNV=train.StartL*2+train.cnv-1
	train.cnv=(train.cnv-0.5)*2
	train.valid=torch.ones(train.state:size(1))
	train.start_loci={}
	train.end_loci={}
	
	train.chrom_state=torch.Tensor(train.state:size(1),1,chrom_width,1);
	train.chrom_state_new=torch.Tensor(train.state:size(1),1,chrom_width,1);
		

	for i=1,train.ChrA:size(1) do
		if (torch.abs(train.Advantage2[i])<3 and torch.rand(1)[1]<0.1/((1+math.exp(-2e-4*counter+8))*(1+math.exp(-train.step[i]+5))) and train.WGD[i]<3) then
			train.state[i]=train.state[i]*2
			train.next[i]=train.next[i]*2
			train.WGD[i]=1
		end
		train.chrom_state[i]=chrom_extract(train.state[i],train.ChrA[i],train.allele[i])
                train.chrom_state_new[i]=train.chrom_state[i]:clone()

		for j=train.StartL[i],chrom_width do
			--train.chrom_state_new[i][1][j][1]=train.chrom_state_new[i][1][j][1]-train.cnv[i]
			if j<=train.End[i] then
				train.chrom_state[i][1][j][1]=train.chrom_state[i][1][j][1]-train.cnv[i]
				train.state[i][train.allele[i]][train.ChrA[i]*chrom_width-chrom_width+j-train.allele[i]*22*chrom_width+22*chrom_width][1]=train.state[i][train.allele[i]][train.ChrA[i]*chrom_width-chrom_width+j-train.allele[i]*22*chrom_width+22*chrom_width][1]-train.cnv[i]
				if(train.chrom_state[i][1][j][1]<0) then
					train.valid[i]=0
				end
			else 
				train.chrom_state_new[i][1][j][1]=train.chrom_state_new[i][1][j][1]+train.cnv[i]
			
			end

		end
		if(train.StartL[i]>1 and torch.abs(train.chrom_state[i][1][train.StartL[i]][1]-train.chrom_state[i][1][train.StartL[i]-1][1])<0.01) then
                        train.valid[i]=0
                end
                if(train.End[i]<50 and torch.abs(train.chrom_state[i][1][train.End[i]][1]-train.chrom_state[i][1][train.End[i]+1][1])<0.01) then
                        train.valid[i]=0
                end
	
		temp_start=torch.zeros(1,chrom_width,1)-1
		temp_copy=train.chrom_state[i]
		temp_start[{{},{2,50},}]:copy(temp_copy[{{},{1,49},}])
		temp_start_loci=(temp_copy-temp_start):select(3,1):nonzero()
		table.insert(train.start_loci,temp_start_loci)

		temp_end=torch.zeros(chrom_width,1)-1
		temp_copy=train.chrom_state[i][1]
		temp_end[{{1,49},}]:copy(temp_copy[{{2,50},}])
		if train.StartL[i]>1 then
				temp_end[{{1,train.StartL[i]-1},}]:copy(temp_copy[{{1,train.StartL[i]-1},}])
		end
		temp_end_loci=(temp_copy-temp_end):select(2,1):nonzero()
		table.insert(train.end_loci,temp_end_loci)
		
	end
	
	--Compute CNP at the next time point
	--Compute Reward-to-go and Advantage
	train.Reward=torch.zeros(train.state:size(1));

	--force WGD
	train.WGD_flag=torch.zeros(train.state:size(1))
	local startL,endL;
	for i=1,train.Reward:size(1) do
		if train.valid[i]>0 then
			train.Reward[i]=Reward(train.ChrA[i],train.StartL[i],train.End[i],train.state[i],train.next[i])
			if((torch.floor(train.next[i]/2)*2-train.next[i]):abs():sum()<1) then
				train.WGD_flag[i]=1
			end
		end
	end
	
		
	--train.Advantage=torch.Tensor(train.state:size(1));
	train.max_cnv=torch.zeros(train.Advantage:size(1)) 
	train.max_end=torch.zeros(train.Advantage:size(1)) 
	train.max_Reward=torch.zeros(train.Advantage:size()) 
	--train.max_end_new=torch.zeros(train.Advantage:size(1)) 
	train.Advantage2=torch.zeros(train.Advantage:size())
	--train.chrom_state_new2=train.chrom_state:clone()
	Advantage_cal();
	--train.Advantage=torch.cmul(train.Advantage,train.valid)
end


LoadData_Reverse=function()
	train.next=train.state:clone()
	
	train.StartL=torch.ones(train.Advantage:size(1))
	train.End=torch.ones(train.Advantage:size(1))*chrom_width

	train.cnv=(train.cnv/2)+0.5
	train.cnv=1-train.cnv

	train.CNV=train.StartL*2+train.cnv-1
	train.cnv=(train.cnv-0.5)*2

	train.valid=torch.ones(train.state:size(1))
	train.start_loci={}
	train.end_loci={}
	
	train.chrom_state=torch.Tensor(train.state:size(1),1,chrom_width,1);
	train.chrom_state_new=torch.Tensor(train.state:size(1),1,chrom_width,1);
		

	for i=1,train.ChrA:size(1) do
		if (torch.rand(1)[1]>0.9 and train.state[i]:sum()<2200*4) then
			--train.state[i]=train.state[i]*2
			--train.next[i]=train.next[i]*2
		end
		train.chrom_state[i]=chrom_extract(train.state[i],train.ChrA[i],train.allele[i])
                train.chrom_state_new[i]=train.chrom_state[i]:clone()

		for j=train.StartL[i],chrom_width do
			--train.chrom_state_new[i][1][j][1]=train.chrom_state_new[i][1][j][1]-train.cnv[i]
			if j<=train.End[i] then
				train.chrom_state[i][1][j][1]=train.chrom_state[i][1][j][1]-train.cnv[i]
				train.state[i][train.allele[i]][train.ChrA[i]*chrom_width-chrom_width+j-train.allele[i]*22*chrom_width+22*chrom_width][1]=train.state[i][train.allele[i]][train.ChrA[i]*chrom_width-chrom_width+j-train.allele[i]*22*chrom_width+22*chrom_width][1]-train.cnv[i]
				if(train.chrom_state[i][1][j][1]<0) then
					train.valid[i]=0
				end
			else 
				train.chrom_state_new[i][1][j][1]=train.chrom_state_new[i][1][j][1]+train.cnv[i]
			
			end

		end
		if(train.StartL[i]>1 and torch.abs(train.chrom_state[i][1][train.StartL[i]][1]-train.chrom_state[i][1][train.StartL[i]-1][1])<0.01) then
                        train.valid[i]=0
                end
                if(train.End[i]<50 and torch.abs(train.chrom_state[i][1][train.End[i]][1]-train.chrom_state[i][1][train.End[i]+1][1])<0.01) then
                        train.valid[i]=0
                end
	
		temp_start=torch.zeros(1,chrom_width,1)-1
		temp_copy=train.chrom_state[i]
		temp_start[{{},{2,50},}]:copy(temp_copy[{{},{1,49},}])
		temp_start_loci=(temp_copy-temp_start):select(3,1):nonzero()
		table.insert(train.start_loci,temp_start_loci)

		temp_end=torch.zeros(chrom_width,1)-1
		temp_copy=train.chrom_state[i][1]
		temp_end[{{1,49},}]:copy(temp_copy[{{2,50},}])
		if train.StartL[i]>1 then
				temp_end[{{1,train.StartL[i]-1},}]:copy(temp_copy[{{1,train.StartL[i]-1},}])
		end
		temp_end_loci=(temp_copy-temp_end):select(2,1):nonzero()
		table.insert(train.end_loci,temp_end_loci)
		
	end
	
	--Compute CNP at the next time point
	--Compute Reward-to-go and Advantage
	train.Reward=torch.zeros(train.state:size(1));

	--force WGD
	train.WGD_flag=torch.zeros(train.state:size(1))
	local startL,endL;
	for i=1,train.Reward:size(1) do
		if train.valid[i]>0 then
			train.Reward[i]=Reward(train.ChrA[i],train.StartL[i],train.End[i],train.state[i],train.next[i])
			if((torch.floor(train.next[i]/2)*2-train.next[i]):abs():sum()<1) then
				train.WGD_flag[i]=1
			end
		end
	end
	
		
	--train.Advantage=torch.Tensor(train.state:size(1));
	train.max_cnv=torch.zeros(train.Advantage:size(1)) 
	train.max_end=torch.zeros(train.Advantage:size(1)) 
	train.max_Reward=torch.zeros(train.Advantage:size()) 
	--train.max_end_new=torch.zeros(train.Advantage:size(1)) 
	train.Advantage2=torch.zeros(train.Advantage:size())
	--train.chrom_state_new2=train.chrom_state:clone()
	Advantage_cal();
	--train.Advantage=torch.cmul(train.Advantage,train.valid)
	train.Advantage2=train.Advantage2

end







