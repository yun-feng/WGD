require "torch"
require "nn"
require "math"

normal_const=5e-5;
single_loci_loss=normal_const*(1-2e-1);
Half_Chromosome_CNV=normal_const*(1-0.4);
Whole_Chromosome_CNV=normal_const*0.99/10;
WGD=normal_const*0.6;
const1=normal_const*(1-1e-1);
const2=2;

Reward=function(ChrA,StartL,EndL,StartS,EndS)
	local reward;
	
		reward=math.log(const1/(const2+math.log(EndL-StartL+1)));
		
			if torch.abs(EndL-StartL+1-chrom_width)<1 then
				reward=math.log(Whole_Chromosome_CNV);
			end

	--		if (StartS==0 and torch.abs(EndS-25.5)<1) or (EndS==50 and torch.abs(StartS-25.5)<1) then
	--			reward=math.log(1.3*Whole_Chromosome_CNV);	
	--		end
		
	return reward;
end

WGD_LOSS=function(cnp,time)
	local reward_next;
	--reward_next=-Chrom_Model:forward((nn.JoinTable(3,3):forward({cnp-1,cnp-2*torch.floor(cnp/2)-1}):resize(1,2,1100,2))):min()
	reward_next=-Chrom_Model:forward(Chrom_input(cnp)):min()
--({cnp,cnp,cnp,torch.floot(cnp/2),torch.floor((cnp+1)/2),train.chr_state,train.chr_next})
	if(reward_next< torch.sum(torch.abs(cnp-1))*math.log(single_loci_loss)) then
		reward_next=torch.sum(torch.abs(cnp-1))*math.log(single_loci_loss)
	end
	
	if (torch.sum(torch.abs(cnp-2*torch.floor(cnp/2))) <1 and torch.sum(cnp)>0) then
		local temp=WGD_LOSS(torch.floor(cnp/2),time)
		if (temp> reward_next) then
			reward_next=temp
			time=time+1
		end
	end
	return reward_next+math.log(WGD),time;
end	

WGD_LOSS_WGD=function(cnp,time)
        local reward_next;
        Chrom_Model:forward(Chrom_input(cnp))
        reward_next=-Chrom_Model:get(11).output:min()
	if(reward_next< torch.sum(torch.abs(cnp-1))*math.log(single_loci_loss)) then
                reward_next=torch.sum(torch.abs(cnp-1))*math.log(single_loci_loss)
        end

        if (torch.sum(torch.abs(cnp-2*torch.floor(cnp/2))) <1 and torch.sum(cnp)>0) then
                local temp=WGD_LOSS(torch.floor(cnp/2),time)
                if (temp> reward_next) then
                        reward_next=temp
                        time=time+1
                end
        end
        return reward_next+math.log(WGD),time;
end

Advantage_cal=function()
	Chrom_Model:forward(Chrom_input(train.next))
--{train.next,train.next,train.next,torch.floot(train.next/2),torch.floor((train.next+1)/2),train.chr_state,train.chr_next})
	train.Advantage=train.Advantage+train.Reward:clone()

	train.max_next=-Chrom_Model.output:min(2):resize(train.Reward:size())
	train.wgd_times=torch.zeros(train.next:size(1))
	for i=1,train.state:size(1) do
			if(torch.sum(torch.abs(train.next[i]-1))*math.log(single_loci_loss) > train.max_next[i]) then
				train.max_next[i]=torch.sum(torch.abs(train.next[i]-1))*math.log(single_loci_loss)
			end
			if(train.WGD_flag[i]==1) then
				local wgd_loss,wgd_time=WGD_LOSS(train.next[i]/2,0)
				if(wgd_loss > train.max_next[i]) then
					train.max_next[i]=wgd_loss
					train.wgd_times[i]=wgd_time

				end
			end
	end
				
	train.Advantage=train.Advantage+torch.cmul(train.max_next,train.valid);
	
	Chrom_Model:forward(Chrom_input(train.state))	
	
	

	CNV_Model:forward(CNV_input(train.chrom_state,train.state))
--{torch.floor(train.state:mean(3):mean(2):expand(train.state:size(1),1,50,1)+0.5),train.chrom_state})
	
	for i = 1,train.state:size(1) do

		if train.valid[i]>0 then
			train.Advantage[i]=train.Advantage[i]+(Chrom_Model.output[i][train.ChrA[i]])

			if train.CNV[i]>1 then
				train.Advantage[i]=train.Advantage[i]-2*(CNV_Model.output[i][train.CNV[i]-1])
			end
			temp_max=CNV_Model.output[i][train.start_loci[i][1][2]*2-1]
			train.max_cnv[i]=train.start_loci[i][1][2]*2-1
			for j=1,train.start_loci[i]:size(1) do
				if(CNV_Model.output[i][train.start_loci[i][j][2]*2-1] > temp_max) then
					temp_max=CNV_Model.output[i][train.start_loci[i][j][2]*2-1]
					train.max_cnv[i]=train.start_loci[i][j][2]*2-1
				end
				if (train.chrom_state[i][1][train.start_loci[i][j][2]][1]-1>0.5) then
					if  (train.start_loci[i][j][2]*2-1>2) and (CNV_Model.output[i][train.start_loci[i][j][2]*2-1-1]>temp_max) then
							temp_max=CNV_Model.output[i][train.start_loci[i][j][2]*2-1-1]
							train.max_cnv[i]=train.start_loci[i][j][2]*2-1-1
					else
						if (train.start_loci[i][j][2]*2-1<2) and (temp_max<0) then
							temp_max=0
							train.max_cnv[i]=0
						end
					end
				end
			end
			train.Advantage[i]=train.Advantage[i]+2*temp_max
			
			--reload max action
			
		end
		
	end
	
	
	End_Point_Model:forward({train.chrom_state,train.chrom_state_new,torch.floor(train.state:mean(3):mean(2):expand(train.state:size(1),1,50,1)+0.5)})
	for i = 1,train.state:size(1) do

		if train.valid[i]>0 then
				
			if train.End[i]>1 then
				train.Advantage[i]=train.Advantage[i]-2*(End_Point_Model.output[i][train.End[i]-1])
			end
			if train.end_loci[i][1][1]>1 then
				temp_max=End_Point_Model.output[i][train.end_loci[i][1][1]-1]
				train.max_end[i]=train.end_loci[i][1][1]
			else
				temp_max=0
				train.max_end[i]=1
			end
			for j=1,train.end_loci[i]:size(1) do
				if(train.cnv[i]>0 or train.chrom_state[i][1][train.end_loci[i][j][1]][1]-1>0.5) then
					if train.end_loci[i][j][1] >1 then
						temp_max=math.max(temp_max,(End_Point_Model.output[i][train.end_loci[i][j][1]-1]))
						train.max_end[i]=train.end_loci[i][1][1]

					else
						temp_max=math.max(temp_max,0)
					end
				end
			end
			train.Advantage[i]=train.Advantage[i]+2*temp_max
		end
	end
	
	train.Advantage2=train.Advantage
	
	return train.Advantage;
end

Advantage_cal2=function()
        Chrom_Model:forward(Chrom_input(train.next))
        train.Advantage=train.Advantage+train.Reward:clone()

        train.max_next=-Chrom_Model.output:min(2):resize(train.Reward:size())
        train.wgd_times=torch.zeros(train.next:size(1))
        for i=1,train.state:size(1) do
                        if(torch.sum(torch.abs(train.next[i]-1))*math.log(single_loci_loss) > train.max_next[i]) then
                                train.max_next[i]=torch.sum(torch.abs(train.next[i]-1))*math.log(single_loci_loss)
                        end
                        if(train.WGD_flag[i]==1) then
                                local wgd_loss,wgd_time=WGD_LOSS(train.next[i]/2,0)
                                if(wgd_loss > train.max_next[i]) then
                                        train.max_next[i]=wgd_loss
                                        train.wgd_times[i]=wgd_time

                                end
                        end
        end

        train.Advantage=train.Advantage+torch.cmul(train.max_next,train.valid);

        Chrom_Model:forward(Chrom_input(train.state))

        CNV_Model:forward(CNV_input(train.chrom_state,train.state))


        for i = 1,train.state:size(1) do

                if train.valid[i]>0 then
                        train.Advantage[i]=train.Advantage[i]+(Chrom_Model.output[i][train.ChrA[i]])
                        train.Advantage2[i]=train.Advantage2[i]+(Chrom_Model.output[i][train.ChrA[i]])

                        if train.CNV[i]>1 then
                                train.Advantage[i]=train.Advantage[i]-2*(CNV_Model.output[i][train.CNV[i]-1])
                        end
                        temp_max=CNV_Model.output[i][train.start_loci[i][1][2]*2-1]
                        train.max_cnv[i]=train.start_loci[i][1][2]*2-1
                        for j=1,train.start_loci[i]:size(1) do
                                if(CNV_Model.output[i][train.start_loci[i][j][2]*2-1] > temp_max) then
                                        temp_max=CNV_Model.output[i][train.start_loci[i][j][2]*2-1]
                                        train.max_cnv[i]=train.start_loci[i][j][2]*2-1
                                end
                                if (train.chrom_state[i][1][train.start_loci[i][j][2]][1]-1>0.5) then
                                        if  (train.start_loci[i][j][2]*2-1>2) and (CNV_Model.output[i][train.start_loci[i][j][2]*2-1-1]>temp_max) then
                                                        temp_max=CNV_Model.output[i][train.start_loci[i][j][2]*2-1-1]
                                                        train.max_cnv[i]=train.start_loci[i][j][2]*2-1-1
                                        else
                                                if (train.start_loci[i][j][2]*2-1<2) and (temp_max<0) then
                                                        temp_max=0
                                                        train.max_cnv[i]=0
                                                end
                                        end
                                end
                        end
                        train.Advantage[i]=train.Advantage[i]+2*temp_max

                        chrom_state_new2=train.chrom_state[i]:clone()
                        temp_start=torch.floor(train.max_cnv[i]/2)+1
                        temp_action=2*((train.max_cnv[i]%2)-0.5)
                        for j=temp_start,chrom_width do
                                chrom_state_new2[1][j][1]=chrom_state_new2[1][j][1]+temp_action
                        end
                        End_Point_Model:forward({train.chrom_state[i],chrom_state_new2,torch.floor(train.state[i]:mean(2):mean(1):expand(1,50,1)+0.5)})

                        temp_end=torch.zeros(chrom_width,1)-1
                        temp_copy=train.chrom_state[i][1]
                        temp_end[{{1,49},}]:copy(temp_copy[{{2,50},}])
                        if temp_start>1 then
                                temp_end[{{1,temp_start-1},}]:copy(temp_copy[{{1,temp_start-1},}])
                        end
                        temp_end_loci=(temp_copy-temp_end):select(2,1):nonzero()
                        if temp_end_loci[1][1]>1 then
                                temp_max=End_Point_Model.output[temp_end_loci[1][1]-1]
                                temp_max_end=temp_end_loci[1][1]
                        else
                                temp_max=0
                                temp_max_end=1
                        end
                        for j=1,temp_end_loci:size(1) do
                                if(temp_action>0 or train.chrom_state[i][1][temp_end_loci[j][1]][1]-1>0.5) then
                                        if (temp_end_loci[j][1] >1) and (End_Point_Model.output[temp_end_loci[j][1]-1]>temp_max) then
                                                                        temp_max=End_Point_Model.output[temp_end_loci[j][1]-1]
                                                                        temp_max_end=temp_end_loci[j][1]
                                        end
                                end
                        end

                        train.next[i]=train.state[i]:clone()
                        for j=temp_start,temp_max_end do
                                train.next[i][train.allele[i]][train.ChrA[i]*chrom_width-chrom_width+j-train.allele[i]*22*chrom_width+22*chrom_width][1]=train.state[i][train.allele[i]][train.ChrA[i]*chrom_width-chrom_width+j-train.allele[i]*22*chrom_width+22*chrom_width][1]+temp_action
                        end
                        train.max_end[i]=temp_max_end
                        train.max_Reward[i]=Reward(train.ChrA[i],temp_start,temp_max_end,train.state[i],train.next[i])
                        train.Advantage2[i]=train.Advantage2[i]+train.max_Reward[i]

                end

        end

        Chrom_Model:forward(Chrom_input(train.next))

        train.max_next=-Chrom_Model.output:min(2):resize(train.Reward:size())
        for i=1,train.state:size(1) do
                        if(torch.sum(torch.abs(train.next[i]-1))*math.log(single_loci_loss) > train.max_next[i]) then
                                train.max_next[i]=torch.sum(torch.abs(train.next[i]-1))*math.log(single_loci_loss)
                        end
                        if((torch.floor(train.next[i]/2)*2-train.next[i]):abs():sum()<1) then
                                local wgd_loss,wgd_time=WGD_LOSS(train.next[i]/2,0)
                                if(wgd_loss > train.max_next[i]) then
                                        train.max_next[i]=wgd_loss
                                end
                        end
        end

        train.Advantage2=train.Advantage2+torch.cmul(train.max_next,train.valid);

        End_Point_Model:forward({train.chrom_state,train.chrom_state_new,torch.floor(train.state:mean(3):mean(2):expand(train.state:size(1),1,50,1)+0.5)})
        for i = 1,train.state:size(1) do

                if train.valid[i]>0 then

                        if train.End[i]>1 then
                                train.Advantage[i]=train.Advantage[i]-2*(End_Point_Model.output[i][train.End[i]-1])
                        end
                        if train.end_loci[i][1][1]>1 then
                                temp_max=End_Point_Model.output[i][train.end_loci[i][1][1]-1]
                        else
                                temp_max=0
                        end
                        for j=1,train.end_loci[i]:size(1) do
                                if(train.cnv[i]>0 or train.chrom_state[i][1][train.end_loci[i][j][1]][1]-1>0.5) then
                                        if train.end_loci[i][j][1] >1 then
                                                temp_max=math.max(temp_max,(End_Point_Model.output[i][train.end_loci[i][j][1]-1]))
                                        else
                                                temp_max=math.max(temp_max,0)
                                        end
                                end
                        end
                        train.Advantage[i]=train.Advantage[i]+2*temp_max
                end
        end

        train.Advantage=train.Advantage-train.Advantage2

        return train.Advantage;
end
