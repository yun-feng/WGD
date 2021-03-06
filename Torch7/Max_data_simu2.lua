require "torch"

require "math"

batch_sample=15
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
		train.state=torch.ones(batch_sample,2,1100,1)
        	train.next=torch.ones(batch_sample,2,1100,1)
		train.Advantage=torch.zeros(batch_sample)
		train.step=torch.zeros(batch_sample)-1
		train.valid=torch.ones(batch_sample)
		train.Advantage2=torch.zeros(batch_sample)
		train.WGD=torch.zeros(batch_sample)
		train.ChrA=torch.floor(torch.rand(train.state:size(1))*(22*2))+1
	end

--	if not flag then
--		train.next=train.state:clone()
--	end
		
		
	--train.ChrA=torch.floor(torch.rand(train.state:size(1))*(22*2))+1;
	train.StartL=torch.floor(torch.rand(train.state:size(1))*chrom_width)+1
	train.End=torch.floor(torch.cmul(torch.rand(train.state:size(1)),(chrom_width-train.StartL+1)))+train.StartL
	--train.allele=torch.floor(torch.rand(train.state:size(1))*2)+1
	for i=1,train.ChrA:size(1) do
		if(torch.rand(1)[1]>0.15+0.83/(1+math.exp(-1e-4*counter+2)) or torch.abs(train.Advantage2[i])>=10) then
			train.state[i]=torch.ones(2,1100,1)
			train.next[i]=torch.ones(2,1100,1)
			train.Advantage[i]=0
			train.step[i]=0
			train.WGD[i]=0
			train.ChrA[i]=torch.floor(torch.rand(1)[1]*(22*2))+1
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
		if(torch.rand(1)[1]>0.8) then
			train.StartL[i]=1
			train.End[i]=chrom_width
		end
		if(torch.rand(1)[1]>0.8) then
			train.ChrA[i]=torch.floor(torch.rand(1)[1]*(22*2))+1
		end
	--	if((not flag) and torch.rand(1)[1]>0.7) then
	--		temp,train.ChrA[i]=Chrom_Model.output[torch.floor(torch.rand(1)[1]*train.state:size(1))+1]:min(1)
	--	end
	end
	
	train.allele=torch.floor((train.ChrA-1)/22)+1
	--train.cnv=(torch.floor(torch.rand(train.state:size(1))*2))
	train.cnv=torch.zeros(train.state:size(1))
	for i=1,train.ChrA:size(1) do
		if(torch.rand(1)[1]>0.5) then
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
		if (torch.abs(train.Advantage2[i])<100 and torch.rand(1)[1]<0.2/(1+math.exp(-train.step[i]+math.min(5,5+torch.sum(train.step)/batch_sample))) and train.WGD[i]<1) then
			train.state[i]=train.state[i]*2
			train.next[i]=train.next[i]*2
			train.WGD[i]=1
			train.cnv[i]=-train.cnv[i]
			if(torch.rand(1)[1]>0.8) then
				train.StartL[i]=1
				train.End[i]=50
				train.cnv[i]=1
			end
			train.CNV[i]=train.StartL[i]*2+torch.floor((train.cnv[i]+1)/2)-1
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
	train.state_cal=nn.JoinTable(3,3):forward({train.state-1,train.state-2*torch.floor(train.state/2)-1})
	 train.next_cal=nn.JoinTable(3,3):forward({train.next-1,train.next-2*torch.floor(train.next/2)-1})
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
		if temp_Advantage2[i]>6 then
			train.valid[i]=0
		end
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
	 train.state_cal=nn.JoinTable(3,3):forward({train.state-1,train.state-2*torch.floor(train.state/2)-1})
         train.next_cal=nn.JoinTable(3,3):forward({train.next-1,train.next-2*torch.floor(train.next/2)-1})
	Advantage_cal();
	--train.Advantage=torch.cmul(train.Advantage,train.valid)
	train.Advantage2=train.Advantage2

end




LoadData_f=function(flag,wgd_f)
        if flag then
                train.state=torch.ones(batch_sample,2,1100,1)
                train.next=torch.ones(batch_sample,2,1100,1)
                train.Advantage=torch.zeros(batch_sample)
                train.step=torch.zeros(batch_sample)-1
                train.valid=torch.ones(batch_sample)
                train.Advantage2=torch.zeros(batch_sample)
                train.WGD=torch.zeros(batch_sample)
        end



        train.ChrA=torch.floor(torch.rand(train.state:size(1))*(22*2))+1;
        train.StartL=torch.floor(torch.rand(train.state:size(1))*chrom_width)+1
        train.End=torch.floor(torch.cmul(torch.rand(train.state:size(1)),(chrom_width-train.StartL+1)))+train.StartL
        for i=1,train.ChrA:size(1) do
                if(torch.rand(1)[1]>0.98/(1+math.exp(-2e-4*counter+2))  ) then
                        train.state[i]=torch.ones(2,1100,1)
                        train.next[i]=torch.ones(2,1100,1)
                        train.Advantage[i]=0
                        train.step[i]=0
                        train.WGD[i]=0
                else
                        if(train.valid[i]>0)  then
                                train.next[i]=train.state[i]:clone()
                                train.step[i]=train.step[i]+1
                                train.Advantage[i]=0
                        else
                                train.state[i]=train.next[i]:clone()
                                train.Advantage[i]=0
                        end

                end
                if(torch.rand(1)[1]>0.7) then
                        train.StartL[i]=1
                        train.End[i]=chrom_width
                end
       
        end

        train.allele=torch.floor((train.ChrA-1)/22)+1
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
                if (wgd_f or (torch.abs(train.Advantage2[i])<3 and torch.rand(1)[1]<0.1/((1+math.exp(-2e-4*counter+2))*(1+math.exp(-train.step[i]+5))) and train.WGD[i]<3)) then
                        train.state[i]=train.state[i]*2
                        train.next[i]=train.next[i]*2
                        train.WGD[i]=1
                end
                train.chrom_state[i]=chrom_extract(train.state[i],train.ChrA[i],train.allele[i])
                train.chrom_state_new[i]=train.chrom_state[i]:clone()

                for j=train.StartL[i],chrom_width do
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

        train.Reward=torch.zeros(train.state:size(1));

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


        train.max_cnv=torch.zeros(train.Advantage:size(1))
        train.max_end=torch.zeros(train.Advantage:size(1))
        train.max_Reward=torch.zeros(train.Advantage:size())
        train.Advantage2=torch.zeros(train.Advantage:size())
        Advantage_cal();
end


LoadData_t=function(step)
	N_batch_sample=20
        train.state=torch.ones(N_batch_sample,2,1100,1)
		train.next=torch.ones(N_batch_sample,2,1100,1)
		train.valid=torch.ones(N_batch_sample)
		train.WGD=torch.zeros(N_batch_sample)


		while(step>0.5) do 
			
			train.ChrA=torch.floor(torch.rand(train.state:size(1))*(22*2))+1;
			train.StartL=torch.floor(torch.rand(train.state:size(1))*chrom_width)+1
			--train.End=torch.floor(torch.cmul(torch.rand(train.state:size(1)),(chrom_width-train.StartL+1)))+train.StartL
			train.End=torch.floor(torch.rand(train.state:size(1))*(chrom_width-1))+train.StartL
			
			train.allele=torch.floor((train.ChrA-1)/22)+1
			train.cnv=torch.zeros(train.state:size(1))
			
			for i=1,train.ChrA:size(1) do
				train.End[i]=math.min(50,train.End[i])
				if(torch.rand(1)[1]>0.6) then
                        train.StartL[i]=1
                        train.End[i]=chrom_width
                end
                if(torch.rand(1)[1]>0.6) then
                        train.cnv[i]=1
                end
			end
			train.CNV=train.StartL*2+train.cnv-1
			train.cnv=(train.cnv-0.5)*2
			
			train.Chr_sample:select(2,step):copy(train.ChrA)
			train.CNV_sample:select(2,step):copy(train.CNV)
			train.End_sample:select(2,step):copy(train.End)
		
			for i=1,train.ChrA:size(1) do
				if(train.valid[i]>0)  then
						train.next[i]=train.state[i]:clone()
				else
						train.state[i]=train.next[i]:clone()
						train.Chr_sample[i][step+1]=0
						train.CNV_sample[i][step+1]=0
						train.End_sample[i][step+1]=0
				end
				train.valid[i]=1
			
                if ((torch.rand(1)[1]<0.1/((1+math.exp(-train.step_sample+step+5))) and train.WGD_sample[i]<0.5) or step==1) then
                    --    train.state[i]=train.state[i]*2
                    --    train.next[i]=train.next[i]*2
                    --    train.WGD_sample[i]=step
                end

                for j=train.StartL[i],train.End[i] do
					train.state[i][train.allele[i]][train.ChrA[i]*chrom_width-chrom_width+j-train.allele[i]*22*chrom_width+22*chrom_width][1]=train.state[i][train.allele[i]][train.ChrA[i]*chrom_width-chrom_width+j-train.allele[i]*22*chrom_width+22*chrom_width][1]-train.cnv[i]
					if(train.state[i][train.allele[i]][train.ChrA[i]*chrom_width-chrom_width+j-train.allele[i]*22*chrom_width+22*chrom_width][1]<0) then
							train.valid[i]=0
					end
				end
				
                if(train.StartL[i]>1 and torch.abs(train.state[i][train.allele[i]][train.ChrA[i]*chrom_width-chrom_width+train.StartL[i]-train.allele[i]*22*chrom_width+22*chrom_width][1]-train.state[i][train.allele[i]][train.ChrA[i]*chrom_width-chrom_width+train.StartL[i]-1-train.allele[i]*22*chrom_width+22*chrom_width][1])<0.01) then
                        train.valid[i]=0
                end
                if(train.End[i]<50 and torch.abs(train.state[i][train.allele[i]][train.ChrA[i]*chrom_width-chrom_width+train.End[i]-train.allele[i]*22*chrom_width+22*chrom_width][1]-train.state[i][train.allele[i]][train.ChrA[i]*chrom_width-chrom_width+train.End[i]+1-train.allele[i]*22*chrom_width+22*chrom_width][1])<0.01) then
                        train.valid[i]=0
                end


        end
	step=step-1
	end
		for i=1,train.ChrA:size(1) do
			if(train.valid[i]<0.5)  then
			train.state[i]=train.next[i]:clone()
			train.Chr_sample[i][step+1]=0
			train.CNV_sample[i][step+1]=0
			train.End_sample[i][step+1]=0
		end
	end

	
end

Deconvolute=function(cnp,max_step)
			flag_chr=0
			flag_cnv=0
			flag_end=0
			rec_flag=false

			current_step=0
			test.ChrA:zero()
			test.CNV:zero()
			test.End:zero()
			while(current_step<max_step) do
				current_step=current_step+1
				cnp_cal=nn.JoinTable(3,3):forward({cnp-1,cnp-torch.floor(cnp/2)*2-1}):resize(1,2,1100,2)
				Chrom_Model:forward(cnp_cal)
				local max_reward,max_chr=Chrom_Model.output:min(2)
				local max_cnv=-1
				local max_end=0
					
				max_reward=-max_reward[1][1]
				max_chr=max_chr[1][1]
				
				if(torch.sum(torch.abs(cnp-2*torch.floor(cnp/2))) <1) then
					local wgd_loss,wgd_time=WGD_LOSS(cnp/2,0)
					if(wgd_loss > max_reward) then
						max_reward=wgd_loss
						max_chr=-1
					end
				end
				
				if(torch.sum(torch.abs(cnp-1))*math.log(single_loci_loss) > max_reward) then
					max_reward=torch.sum(torch.abs(cnp-1))*math.log(single_loci_loss)
					max_chr=0
					current_step=max_step
				end
				
				if max_chr>0 then
					local max_allele=torch.floor((max_chr-1)/22)+1
					chrom_state=chrom_extract(cnp,max_chr,max_allele)
					temp_start=torch.zeros(1,chrom_width,1)-1
					temp_copy=chrom_state:clone()
					temp_start[{{},{2,50},}]:copy(temp_copy[{{},{1,49},}])
					temp_start_loci=(temp_copy-temp_start):select(3,1):nonzero()
					
					
					
					CNV_Model:forward({torch.floor(cnp:mean(2):mean(1):expand(1,50,1)+0.5),chrom_state})
					
					temp_max=CNV_Model.output[temp_start_loci[1][2]*2-1]
					max_cnv=temp_start_loci[1][2]*2-1
					
					for j=1,temp_start_loci:size(1) do
						
						if(CNV_Model.output[temp_start_loci[j][2]*2-1] > temp_max) then
							temp_max=CNV_Model.output[temp_start_loci[j][2]*2-1]
							max_cnv=temp_start_loci[j][2]*2-1
						end
						if (chrom_state[1][temp_start_loci[j][2]][1]-1>-0.5 and not (rec_flag and flag_cnv==temp_start_loci[j][2]*2-1-1)) then
							if  (temp_start_loci[j][2]*2-1>2) and (CNV_Model.output[temp_start_loci[j][2]*2-1-1]>temp_max) then
									temp_max=CNV_Model.output[temp_start_loci[j][2]*2-1-1]
									max_cnv=temp_start_loci[j][2]*2-1-1
							else
								if (temp_start_loci[j][2]*2-1<2) and (temp_max<0) then
									temp_max=0
									max_cnv=0
								end
							end
						end
					end
				
					
					
					
					chrom_state_new=chrom_state:clone()
					temp_start=torch.floor(max_cnv/2)+1
					temp_action=2*((max_cnv%2)-0.5)
					for j=temp_start,chrom_width do
						chrom_state_new[1][j][1]=chrom_state_new[1][j][1]+temp_action
					end
					End_Point_Model:forward({chrom_state,chrom_state_new,torch.floor(cnp:mean(2):mean(1):expand(1,50,1)+0.5)})
					
					temp_end=torch.zeros(chrom_width,1)-1
					temp_copy=chrom_state[1]
					temp_end[{{1,49},}]:copy(temp_copy[{{2,50},}])
					if temp_start>1 then
						temp_end[{{1,temp_start-1},}]:copy(temp_copy[{{1,temp_start-1},}])
					end
					temp_end_loci=(temp_copy-temp_end):select(2,1):nonzero()
					if temp_end_loci[1][1]>1 then
						temp_max=End_Point_Model.output[temp_end_loci[1][1]-1]
						max_end=temp_end_loci[1][1]
					else
						temp_max=0
						max_end=1
					end
					for j=1,temp_end_loci:size(1) do
						if(temp_action>0 or chrom_state[1][temp_end_loci[j][1]][1]-1>-0.5) then
							if (temp_end_loci[j][1] >1) and (End_Point_Model.output[temp_end_loci[j][1]-1]>temp_max) then
											temp_max=End_Point_Model.output[temp_end_loci[j][1]-1]
											max_end=temp_end_loci[j][1]
							end
						elseif (temp_action<0 and chrom_state[1][temp_end_loci[j][1]][1]-1<-0.5) then
							break
						end
					end
					
					for j=temp_start,max_end do
						cnp[max_allele][max_chr*chrom_width-chrom_width+j-max_allele*22*chrom_width+22*chrom_width][1]=cnp[max_allele][max_chr*chrom_width-chrom_width+j-max_allele*22*chrom_width+22*chrom_width][1]+temp_action
					end
				
				
				end
				
				if max_chr<0 then
					cnp:copy(cnp/2)
				end
			
				test.ChrA[current_step]=max_chr
				test.CNV[current_step]=max_cnv+1
				test.End[current_step]=max_end
					if(max_cnv%2==1 and flag_chr==max_chr and flag_cnv==max_cnv and flag_end==max_end) then
						rec_flag=true
						flag_cnv=2*torch.floor(max_cnv/2)
					else
						rec_flag=false
						flag_chr=max_chr
						flag_cnv=(1-max_cnv%2)+2*torch.floor(max_cnv/2)
						flag_end=max_end
					end
				
			end
end


