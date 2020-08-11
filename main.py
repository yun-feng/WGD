#main.py
import torch
import math
import torch.optim as optim
#import Policy
#import Data_train

#setting up counter
counter_global=0
#Model
Q_model=Q_learning()
#Load initial data
state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,wgd,step,advantage,valid=Load_data()
#setting up optimizer
optimizer = optim.Adam(Q_model.parameters(), lr=1e-3,betas=(0.9, 0.99), eps=1e-08, weight_decay=1e-6)


#start training
while(counter_global< 1e6):
  counter_global=counter_global+1
  #load data
  state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,wgd,step,advantage,valid=Load_data(False,state,next_state,advantage,Chr,step,wgd,valid)
  #compute advantage
  optimizer.zero_grad()
  advantage,cnv_max,sigma,temp,t2,t3=Q_model.forward(state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,advantage,valid)
  #compute loss
  loss=advantage.pow(2).mean()
  if(counter_global%10==0):
    print(loss)
    #print(temp[0])
    print(step.mean())
  #train the model
  loss.backward()
  optimizer.step()
  
  
  #training with the best action
  #temp for the values both used in training and loading new data
  state_temp,next_state_temp,chrom,chrom_new,cnv,start_loci,end_loci,advantage_temp=Modify_data(state,chrom,Chr,valid,cnv_max,Q_model.End,sigma)
  #compute advantage
  optimizer.zero_grad()
  advantage,cnv_max,sigma,temp,temp2,temp3=Q_model.forward(state_temp,next_state_temp,chrom,chrom_new,Chr,cnv,start_loci,end_loci,advantage_temp,valid)
  #compute loss
  loss=advantage.pow(2).mean()
  
  ##debugging purpose
  if(loss> 1e8):
    print(advantage)
    print(advantage[torch.gt(advantage,1e8)])
    print(temp[torch.gt(advantage,1e8)])
    print(temp2[torch.gt(advantage,1e8)])
    print(temp3[torch.gt(advantage,1e8)])
    print(chrom[torch.gt(advantage,1e8)])
    print(chrom_new[torch.gt(advantage,1e8)])
    print(start_loci[torch.gt(advantage,1e8)])
    print(end_loci[torch.gt(advantage,1e8)])
    print(cnv[torch.gt(advantage,1e8)])
    break
  
  if(counter_global%10==0):
    print(loss)
    #print(temp[0])
  #train the model
  loss.backward()
  optimizer.step()