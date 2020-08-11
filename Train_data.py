#Train_data.py
import torch
import math

batch_size=15
print(batch_size)
#during training
#data are simulated backwards
#when step==0, it means it is the last step for the trajectory
#and step++ to make CNP more complex
def Load_data(first_step_flag=True,state=None,next_state=None,advantage=None,Chr=None,step=None,wgd=None,valid=None):
  if first_step_flag:
    state=torch.ones(batch_size,1,44,50,requires_grad=False)
    next_state=torch.ones(batch_size,1,44,50,requires_grad=False)
    Chr=torch.ones(batch_size,requires_grad=False).type(torch.LongTensor)
    step=torch.zeros(batch_size,requires_grad=False)
    advantage=torch.zeros(batch_size)
    wgd=torch.zeros(batch_size,requires_grad=False)
    valid=torch.ones(batch_size,requires_grad=False)
    
  start_loci=torch.randint(high=50,size=(batch_size,),requires_grad=False)
  end_loci=torch.LongTensor(batch_size)
  cnv=torch.ones(batch_size,requires_grad=False)
  chrom=torch.Tensor(batch_size,50)
  chrom_new=torch.Tensor(batch_size,50)
  #probability of resetting the training trajectory back to step=0
  step_prob=0.15+0.4/(1+math.exp(-1e-4*counter_global+2))
  for i in range(batch_size):
    #if the model is poorly trained until the current step
    #go back to the state 0
    #to ensure small error for short trajectories
    if(torch.rand(1)[0]>step_prob or torch.abs(advantage[i])>=15):
      state[i]=torch.ones(1,44,50,requires_grad=False)
      next_state[i]=torch.ones(1,44,50,requires_grad=False)
      step[i]=0
    #if model is fully trained for the current step
    #go to next step
    elif(valid[i]>0 and torch.abs(advantage[i])<7):
      next_state[i]=state[i].clone()
      step[i]=step[i]+1
    #stay to further train the current step
    else:
      state[i]=next_state[i].clone()
      
    advantage[i]=0
    end_loci[i]=1+torch.randint(low=start_loci[i],high=50,size=(1,))[0]
    #change the chromosone that CNV is on with some probability
    if torch.rand(1)[0]>0.5:
      Chr[i]=torch.randint(high=44,size=(1,))[0]
    #adding probability to sample chromosomal changes during training
    if torch.rand(1)[0]>0.8:
      start_loci[i]=0
      end_loci[i]=50
    #cnv
    if torch.rand(1)[0]>0.7:
      cnv[i]=0
    #modifying cnp
    prob_wgd=0.4/(1+math.exp(-step[i]+5))
    #wgd          
    if (torch.abs(advantage[i])<100 and torch.rand(1)[0]<prob_wgd and wgd[i]<1):
      wgd[i]=1
      state[i]=state[i]*2
      next_state[i]=next_state[i]*2
    #adding cnv effect
    #increasing copies when no wgd
    #decreasing copies when wgd
    if wgd[i]>0.5:
      cnv[i]=1-cnv[i]
    state[i][0][Chr[i]][(start_loci[i]):(end_loci[i])]=state[i][0][Chr[i]][(start_loci[i]):(end_loci[i])]-(cnv[i]-0.5)*2
    chrom[i]=state[i][0][Chr[i]][:]
    #reverse effect on chrom_new
    chrom_new[i]=state[i][0][Chr[i]][:]
    chrom_new[i][(start_loci[i]):]=chrom_new[i][(start_loci[i]):]+(cnv[i]-0.5)*2
    #not going to negative values
    if(torch.any(state[i][0][Chr[i]][(start_loci[i])]< -0.5)):
      valid[i]=0
    #not joining breakpoints
    if(start_loci[i]>0.5 and torch.abs(chrom[i][start_loci[i]]-chrom[i][start_loci[i]-1])<0.5):
      valid[i]=0
    if(end_loci[i]<49.5 and torch.abs(chrom[i][end_loci[i]-1]-chrom[i][end_loci[i]])<0.5):
      valid[i]=0
  return state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,wgd,step,advantage,valid

def Modify_data(state,chrom,Chr,valid,cnv_max,model,sigma):
  #Modify the training data to train the Q values for the best action
  #place takers
  #make sure they are of correct tensor types
  #make sure they are meaningful values to avoid inf if they are not valid samples
  #otherwise nan may be generated
  start_loci=torch.randint(high=50,size=(batch_size,),requires_grad=False)
  end_loci=start_loci.clone()
  cnv=torch.ones(batch_size,requires_grad=False)
  next_state=state.clone()
  chrom_new=chrom.clone()
  advantage=torch.zeros(batch_size)
  for i in range(batch_size):
    #only deal with valid samples
    if valid[i]>0.5:
      start_loci[i]=cnv_max[i]//2
      cnv[i]=cnv_max[i]-start_loci[i]*2
      #update chrom_new
      chrom_new[i][(start_loci[i]):]=chrom_new[i][(start_loci[i]):]+(cnv[i]-0.5)*2
  
  end_loci=model.find_end(chrom,chrom_new,sigma,start_loci,cnv,valid)
  
  for i in range(batch_size):
      next_state[i][0][Chr[i]][(start_loci[i]):(end_loci[i])]=next_state[i][0][Chr[i]][(start_loci[i]):(end_loci[i])]+(cnv[i]-0.5)*2
      
      
  return state,next_state,chrom,chrom_new,cnv,start_loci,end_loci,advantage

if __name__ == "__main__":
  counter_global=torch.randint(10000,(1,))[0]
  state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,wgd,step,advantage,valid=Load_data()
  """
  print(Chr)
  print(cnv)
  print(start_loci)
  print(end_loci)
  print(wgd)
  print(step)
  print(advantage)
  print(state.shape)
  print(next_state.shape)
  print(state[0][0][Chr[0]])
  print(next_state[0][0][Chr[0]])
  print(chrom[0])
  print(chrom_new[0])
  """
  res,cnv_max,sigma,t,t2,t3=Q_model.forward(state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,advantage,valid)
  print(valid)
  print(res)
  print(cnv_max)
  loss=res.pow(2).mean()
  print(loss)
  loss.backward()
  params = list(Q_model.parameters())
  print(params[0].grad[0])
  print(Q_model.switch.conv1.weight[0])
  state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,wgd,step,advantage,valid=Load_data(False,state,next_state,advantage,Chr,step,wgd,valid)
  """
  print(Chr)
  print(cnv)
  print(start_loci)
  print(end_loci)
  print(wgd)
  print(step)
  print(advantage)
  print(state.shape)
  print(next_state.shape)
  print(state[0][0][Chr[0]])
  print(next_state[0][0][Chr[0]])
  print(chrom[0])
  print(chrom_new[0])
  """
  res,cnv_max,sigma,t,t2,t3=Q_model.forward(state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,advantage,valid)
  print(res)
  print(cnv_max)
  loss=res.pow(2).mean()
  print(loss)
  loss.backward()
  params = list(Q_model.parameters())
  print(params[0].grad[0])
  print(Q_model.switch.conv1.weight[1])