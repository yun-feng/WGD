#Policy.py
import torch
import math
from torch import nn
import torch.nn.functional as F


chrom_width=50;
nkernels = [160,240,480]

normal_const=5e-5;
single_loci_loss=normal_const*(1-2e-1);
WGD=normal_const*0.6;

#switch
class WGD_Net(nn.Module):
  def __init__(self):
    super(WGD_Net, self).__init__()
    #chromosome permutation invariant
    #slide for chromosome is 1 and the filter length in this dimension is also 1
    #thus, the same filter goes through all chromosomes in the same fashion
    self.conv1=nn.Conv2d(1, nkernels[0]//4, (1,3),(1,1),(0,1))
    self.conv2=nn.Conv2d(nkernels[0]//4,nkernels[1]//4 , (1,3),(1,1), (0,1))
    self.conv3=nn.Conv2d(nkernels[1]//4,nkernels[2]//4 , (1,5),(1,1), (0,0))
    self.linear=nn.Linear(nkernels[2]//4,1)
    
  def forward(self, x):
    y=x.mean((1,2,3))
    y=y.reshape(x.shape[0],1)
    x=x.reshape(x.shape[0],1,44,50)
    x=F.max_pool2d(F.relu(self.conv1(x)),(1,5),(1,5),(0,0))
    x=F.max_pool2d(F.relu(self.conv2(x)),(1,2),(1,2),(0,0))
    x=(F.relu(self.conv3(x))).sum((2,3))
    x=self.linear(x)
    #residule representation in x
    x=20*(y-1.5)+x
    x=torch.sigmoid(x)
    return x

#chromosome evaluation net
class CNP_Val(nn.Module):
  def __init__(self):
    super(CNP_Val, self).__init__()
    self.conv1=nn.Conv2d(1, nkernels[0]//2, (1,5),(1,1),(0,2))
    self.conv2=nn.Conv2d(nkernels[0]//2,nkernels[1]//2 , (1,3),(1,1), (0,1))
    self.conv3=nn.Conv2d(nkernels[1]//2,nkernels[2]//3 , (1,3),(1,1), (0,1))
    self.conv4=nn.Conv2d(nkernels[2]//3,1, (1,5),(1,1), (0,0))
    self.linear=nn.Linear(nkernels[2]//4,1)
    
  def forward(self, x):
    x=F.max_pool2d(F.relu(self.conv1(x)),(1,3),(1,3),(0,1))
    x=F.max_pool2d(F.relu(self.conv2(x)),(1,2),(1,2),(0,1))
    x=F.max_pool2d(F.relu(self.conv3(x)),(1,2),(1,2),(0,1))
    #KL divergence is always nonpositive
    x=0.25+F.elu(self.conv4(x),0.25)
    x=x.reshape(x.shape[0],44)
    return x
  
class Chrom_NN(nn.Module):
  def __init__(self):
    super(Chrom_NN,self).__init__()
    #two parts of the Chrom_NN
    #NN for CNP without WGD 
    self.Val_noWGD=CNP_Val()
    #NN for CNP with WGD
    self.Val_WGD=CNP_Val()
    
  def forward(self,x,sigma):
    #probability for WGD
    sigma=sigma.expand(-1,44)
    #residule representation for reward-to-go of CNP
    #we assume the copy number for each loci ranges from 0~9
    #for samples without WGD
    y=torch.ceil((torch.abs(x-1)).mean(3)/12)
    y=y.reshape(x.shape[0],44)
    Val_no=self.Val_noWGD.forward(x)
    Val_no=y*math.log(single_loci_loss)*2+(((1-y)*Val_no).sum(1)).reshape(x.shape[0],1).expand(-1,44)
    #for samples with WGD
    z=torch.ceil((torch.abs(x-2*(x//2))).sum(3)/100)
    z=z.reshape(x.shape[0],44)
    Val_wgd=self.Val_WGD.forward(x)
    Val_wgd=z*math.log(single_loci_loss)*2+((1-z)*Val_wgd).sum(1).reshape(x.shape[0],1).expand(-1,44)
    #Val_wgd=Val_wgd+math.log(WGD)
    #combine two NN with switch
    x=sigma*Val_wgd+(1-sigma)*Val_no
    x=-x
    return x



#starting point and gain or loss 
class CNV_Val(nn.Module):
  def __init__(self):
    super(CNV_Val,self).__init__()
    self.conv1=nn.Conv2d(1, nkernels[0]//2, (1,7),(1,1),(0,3))
    self.conv2=nn.Conv2d(nkernels[0]//2,nkernels[1]//2 , (1,7),(1,1), (0,3))
    self.conv3=nn.Conv2d(nkernels[1]//2,nkernels[2]//3 , (1,7),(1,1), (0,3))
    self.conv4=nn.Conv2d(nkernels[2]//3, 10, (1,7),(1,1), (0,3))
    self.linear=nn.Linear(10*50,99)

  def forward(self,x):
    x=F.relu(self.conv1(x))
    x=F.relu(self.conv2(x))
    x=F.relu(self.conv3(x))
    x=F.relu(self.conv4(x))
    x=x.reshape(x.shape[0],10*50)
    x=self.linear(x)
    return x

class CNV_NN(nn.Module):
  def __init__(self):
    super(CNV_NN,self).__init__()
    #two network setting
    self.CNV_noWGD=CNV_Val()
    self.CNV_WGD=CNV_Val()
    
  def forward(self,x,sigma):
    #residule representation for noWGD
    y=torch.Tensor(x.shape[0],50,2)
    y[:,:,0]=F.relu(1-x)
    y[:,:,1]=F.relu(x-1)
    y=y.reshape(x.shape[0],100)
    y=y[:,1:100]-y[:,0:1].expand(-1,99)
    Val_no=self.CNV_noWGD.forward(x.reshape(x.shape[0],1,1,50))
    Val_no=y+Val_no
    #residule representation for WGD
    z=((torch.abs(x-2*(x//2))).reshape(x.shape[0],50,1)).expand(-1,-1,2)
    z=z.reshape(x.shape[0],100)
    z=z[:,1:100]-z[:,0:1].expand(-1,99)
    Val_wgd=self.CNV_WGD.forward(x.reshape(x.shape[0],1,1,50))
    Val_wgd=z+Val_wgd
    #switch
    x=sigma*Val_wgd+(1-sigma)*Val_no
    return(x)
    


#end point
class End_Point_Val(nn.Module):
  def __init__(self):
    super(End_Point_Val,self).__init__()
    self.conv1=nn.Conv2d(2, nkernels[0]//2, (1,7),(1,1),(0,3))
    self.conv2=nn.Conv2d(nkernels[0]//2,nkernels[1]//2 , (1,7),(1,1), (0,3))
    self.conv3=nn.Conv2d(nkernels[1]//2,nkernels[2]//2 , (1,7),(1,1), (0,3))
    self.linear=nn.Linear(nkernels[2]//2*50,49)
  
  def forward(self,old,new):
    x=torch.Tensor(old.shape[0],2,1,50)
    x[:,0,0,:]=old
    x[:,1,0,:]=new
    x=F.relu(self.conv1(x))
    x=F.relu(self.conv2(x))
    x=F.relu(self.conv3(x))
    x=x.reshape(x.shape[0],nkernels[2]//2*50)
    x=self.linear(x)
    return x
    
class End_Point_NN(nn.Module):
  def __init__(self):
    super(End_Point_NN,self).__init__()
    #two network setting
    self.Val_noWGD=End_Point_Val()
    self.Val_WGD=End_Point_Val()
  
  def forward(self,old,new,sigma):
    #residule representation for noWGD
    y=F.relu((old-1)*(old-new))
    y=y[:,1:50]-y[:,0:1].expand(-1,49)
    Val_no=self.Val_noWGD.forward(old,new)
    Val_no=Val_no+y
    #residule representation for WGD
    z=(old-2*(old//2))*(1-(new-2*(new//2)))
    z=z[:,1:50]-z[:,0:1].expand(-1,49)
    Val_wgd=self.Val_WGD.forward(old,new)
    Val_wgd=Val_wgd+z
    #switch
    x=sigma*Val_wgd+(1-sigma)*Val_no
    return x
  
  #used for finding the end during deconvolution
  def find_end(self,old,new,sigma,start_loci,cnv,valid):
    res_end=self.forward(old,new,sigma)
    
    break_end=torch.zeros(state.shape[0],50,requires_grad=False)
    chrom_shift=torch.zeros(state.shape[0],50,requires_grad=False)
    chrom_shift[:,:49]=chrom[:,1:]
    #allow adding one copy for every breakpoint
    break_end[:,:]=torch.ceil(torch.abs(chrom-chrom_shift)/10)
    #always allow adding one chromosone
    break_end[:,49]=1
    
    for i in range(old.shape[0]):
      #can't end before starting point
      break_end[i,:int(start_loci[i])]=0*break_end[i,:int(start_loci[i])]
      #don't allow lose one copy when copy number equalls 0
      if(cnv[i]<0.5):
        j=int(start_loci[i])+1
        while(j<50):
          if(chrom[i][j]<0.5):
            break
          j=j+1
        break_end[i,j:50]=0*break_end[i,j:50]
    res_end_full=torch.zeros(state.shape[0],50)
    res_end_full[:,1:]=res_end
    #Prior_rule=break_end
    res_end_full=res_end_full+torch.log(break_end)
    end_max_val,end_max=torch.max(res_end_full,1)
    return end_max+1
        

#combine all separate networks
#add Rule system

#calculating the softmax
#prevent inf when taking log(exp(x))
#log_exp is always gonna be between 1 and the total number of elements
def Soft_update(val1,soft1,val2,soft2):
  bias=val1.clone()
  log_exp=soft1.clone()
  set1=[torch.ge(val1,val2)]
  bias[set1]=val1[set1]
  log_exp[set1]=soft1[set1]+soft2[set1]*torch.exp(val2[set1]-val1[set1])
  set2=[torch.lt(val1,val2)]
  bias[set2]=val2[set2]
  log_exp[set2]=soft2[set2]+soft1[set2]*torch.exp(val1[set2]-val2[set2])
  return bias,log_exp


#reward
#log probability of CNV
const1=normal_const*(1-1e-1);
const2=2;
Whole_Chromosome_CNV=normal_const*0.99/10;
Half_Chromosome_CNV=normal_const*0.99/15;

def Reward(Start,End):
  Start=Start.to(torch.float32)
  End=End.to(torch.float32)
  reward=torch.log(const1/(const2+torch.log(End-Start)))
  #chromosome changes
  for i in range(Start.shape[0]):
    #full chromosome
    if End[i]-Start[i]>49.5:
      reward[i]=math.log(Whole_Chromosome_CNV)
    #arm level changes
    if 50-End[i]<0.5 and abs(25-Start[i])<1.5:
      reward[i]=math.log(Half_Chromosome_CNV)
    if Start[i]<0.5 and abs(25-End[i])<1.5:
      reward[i]=math.log(Half_Chromosome_CNV)
  return reward

class Q_learning(nn.Module):
  def __init__(self):
    super(Q_learning,self).__init__()
    self.switch=WGD_Net()
    self.Chrom_model=Chrom_NN()
    self.CNV=CNV_NN()
    self.End=End_Point_NN()
  def Softmax(self,next_state,sigma):
    x=self.Chrom_model.forward(next_state,sigma)
    max_chrom=torch.max(x,1)[0]
    softmax_chrom=x-max_chrom.reshape(x.shape[0],1).expand(-1,44)
    softmax_chrom=torch.exp(softmax_chrom).sum(1)
    #special action END
    end_val=torch.sum(torch.abs(next_state-1),(1,2,3))*math.log(single_loci_loss)
    max_chrom,softmax_chrom=Soft_update(max_chrom,softmax_chrom,end_val,torch.ones(x.shape[0]))
    #if there is a WGD followed immediately
    #rule system
    for i in range(x.shape[0]):
      if (not torch.any(next_state[i]-2*torch.floor(next_state[i]/2)>0.5)) and torch.any(next_state[i]>0.5):
        sigma_wgd=self.switch(torch.floor(next_state[i:(i+1)]/2))
        sigma_wgd=sigma_wgd.detach()
        wgd_val,wgd_soft=self.Softmax(torch.floor(next_state[i:(i+1)]/2),sigma_wgd)
        max_chrom[i],softmax_chrom[i]=Soft_update(torch.ones(1)*max_chrom[i],torch.ones(1)*softmax_chrom[i],torch.ones(1)*wgd_val,torch.ones(1)*wgd_soft)
    #in Q-learning
    #gradient does not flow through the softmax part in order to mimic regression problems
    max_chrom=max_chrom.detach()
    softmax_chrom=softmax_chrom.detach()
    return max_chrom,softmax_chrom
  
  def forward(self,state,next_state,chrom,chrom_new,Chr,cnv,start_loci,end_loci,advantage,valid):
    #moving target as constant
    sigma_next=self.switch(next_state)
    sigma_next=sigma_next.detach()
    x,y=self.Softmax(next_state,sigma_next)
    x=x+torch.log(y)
    x=x+Reward(start_loci,end_loci)
    #Q(s,a)
    sigma=self.switch.forward(state)
    res_chrom=self.Chrom_model.forward(state,sigma)
    res_cnv=self.CNV.forward(chrom,sigma)
    #if there is originally a break point for start
    #rule system
    break_start=torch.zeros(state.shape[0],50,2,requires_grad=False)
    chrom_shift=torch.zeros(state.shape[0],50,requires_grad=False)
    chrom_shift[:,1:]=chrom[:,:49]
    #allow adding one copy for every breakpoint
    break_start[:,:,1]=torch.ceil(torch.abs(chrom-chrom_shift)/10)
    #always allow adding one chromosone
    break_start[:,0,1]=1
    #don't allow lose one copy when copy number equalls 0
    break_start[:,:,0]=break_start[:,:,1]
    break_start[:,:,0]=break_start[:,:,0]*torch.ceil(chrom/10)
    break_start=break_start.reshape(state.shape[0],100)
    res_cnv_full=torch.zeros(state.shape[0],100)
    res_cnv_full[:,1:]=res_cnv
    #Prior_rule=break_start
    res_cnv_full=res_cnv_full+torch.log(break_start)
    #best cnv according to the current Q
    cnv_max_val,cnv_max=torch.max(res_cnv_full,1)
    cnv_softmax=res_cnv_full-cnv_max_val.reshape(state.shape[0],1).expand(-1,100)
    cnv_softmax=torch.exp(cnv_softmax).sum(1)
    x=x+cnv_max_val+torch.log(cnv_softmax)
    
    res_end=self.End.forward(chrom,chrom_new,sigma)
    #if there is originally a break point for end
    #and if this is after the starting point
    #rule system
    break_end=torch.zeros(state.shape[0],50,requires_grad=False)
    chrom_shift=torch.zeros(state.shape[0],50,requires_grad=False)
    chrom_shift[:,:49]=chrom[:,1:]
    #allow adding one copy for every breakpoint
    break_end[:,:]=torch.ceil(torch.abs(chrom-chrom_shift)/10)
    #always allow adding one chromosone
    break_end[:,49]=1
    
    for i in range(state.shape[0]):
      #can't end before starting point
      break_end[i,:int(start_loci[i])]=0*break_end[i,:int(start_loci[i])]
      #don't allow lose one copy when copy number equalls 0
      if(cnv[i]<0.5):
        j=int(start_loci[i])+1
        while(j<50):
          if(chrom[i][j]<0.5):
            break
          j=j+1
        break_end[i,j:50]=0*break_end[i,j:50]
    res_end_full=torch.zeros(state.shape[0],50)
    res_end_full[:,1:]=res_end
    #Prior_rule=break_end
    res_end_full=res_end_full+torch.log(break_end)
    end_max_val,end_max_temp=torch.max(res_end_full,1)
    end_softmax=res_end_full-end_max_val.reshape(state.shape[0],1).expand(-1,50)
    end_softmax=torch.exp(end_softmax).sum(1)
    x=x+end_max_val+torch.log(end_softmax)
    for i in range(state.shape[0]):
      if valid[i]>0.5:#check validity to prevent inf-inf which ends in nan
        x[i]=x[i]-res_chrom[i][int(Chr[i])]
        cnv_rank=int(start_loci[i]*2+cnv[i])
        x[i]=x[i]-res_cnv_full[i][cnv_rank]
        end_rank=int(end_loci[i]-1)
        x[i]=x[i]-res_end_full[i][end_rank]
    
    #remove training data which include invalid actions
    x=x*valid
    #return avdantage as well as a best cnv and sigma used for generating training data
    #used for training in the next step
    return x,cnv_max,sigma,res_chrom,res_cnv_full,res_end_full
  

if __name__ == "__main__":
  #test different parts separately
  '''
  switch=WGD_Net()
  Chrom_model=Chrom_NN()
  print(Chrom_model)
  #test the structure of permutation invariant structure
  x=torch.ones(3,1,44,50)
  x[0][0][0][0:50]=2
  x[2][0][1][0:50]=2
  prob=switch.forward(x)
  print(prob)
  res=Chrom_model.forward(x,prob)
  print(res)
  res=-float('inf')
  res=torch.LongTensor(3)
  
  print(torch.log(res.type(torch.DoubleTensor)))
  #CNV
  CNV=CNV_NN()
  res=CNV.forward(x[:,0,0,0:50],prob)
  print(CNV)
  print(res.shape)
  #END
  End=End_Point_NN()
  res=End.forward(x[:,0,0,0:50],x[:,0,0,0:50]+1,prob)
  print(End)
  print(res.shape)
  '''
  #test Q-learning
  x=torch.ones(3,1,44,50)
  y=torch.ones(3,1,44,50)
  x[0][0][0][0:50]=2
  x[2][0][1][0:50]=2
  chrom=x[:,0,0,:]
  chrom_new=y[:,0,0,:]
  Chr=torch.zeros(3)
  cnv=torch.zeros(3)
  start_loci=torch.zeros(3)
  end_loci=torch.ones(3)*50
  advantage=torch.zeros(3)
  valid=torch.ones(3)
  Q_model=Q_learning()
  res,cnv_max,sigma,t,t2,t3=Q_model.forward(x,y,chrom,chrom_new,Chr,cnv,start_loci,end_loci,advantage,valid)
  print(res)
  print(cnv_max)
  loss=res.pow(2).mean()
  print(loss)
  loss.backward()
  params = list(Q_model.parameters())
  print(params[0].grad[0])
  print(Q_model.switch.conv1.weight[0])


