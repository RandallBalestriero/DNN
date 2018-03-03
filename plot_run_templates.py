import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 19
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size
from scipy.io import savemat
from tabulate import tabulate

def loadfile(name):
        f=open(name,'rb')
        data=cPickle.load(f)
        f.close()
        templates = data[0][0]
        x = data[0][1]
	y = data[0][2]
#        x-=x.min((1,2,3),keepdims=True)
#        x/=x.max((1,2,3),keepdims=True)
#        templates-=templates.min((2,3,4),keepdims=True)
#        templates/=templates.max((2,3,4),keepdims=True)
#	test = data[0][2]
        predsy1 = data[1][0]
        preds1  = data[1][1]
        Ax1     = data[1][2]
        bx1     = data[1][3]
        predsy2 = data[2][0]
        preds2  = data[2][1]
        Ax2     = data[2][2]
        bx2     = data[2][3]
	return x,y,templates,predsy1,preds1,Ax1,bx1,predsy2,preds2,Ax2,bx2

cifarclass = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def subplots(x,y,templates,predsy1,preds1,Ax1,bx1,predsy2,preds2,Ax2,bx2,offset,k,cropped):
	rx=x-x.min((1,2,3),keepdims=True)
	rx/=rx.max((1,2,3),keepdims=True)
	rtemplates=templates-templates.min((1,2,3,4),keepdims=True)
	rtemplates/=rtemplates.max((1,2,3,4),keepdims=True)
        d=dict()
        z=[x[k]]
	if(cropped==0):
        	subplot(4,11,1+offset)
	else:
                subplot(1,4,1+offset)
        if(x.shape[-1]==1):
                imshow(rx[k,:,:,0],cmap='gray',interpolation='nearest')
		q=[y[k]]
	else:
                imshow(rx[k],interpolation='nearest')
		q=[cifarclass[y[k]]]
	if(offset==0):
		title('Input',fontsize=20)
        xticks([])
        yticks([])
	if(cropped==0):
	        for i in xrange(10):
	                subplot(4,11,2+i+offset)
	                if(x.shape[-1]==1 or len(Ax1)>55000):
	                        imshow(rtemplates[k,i,:,:,0],cmap='gray',interpolation='nearest',vmin=0,vmax=1)
			else:
	                        imshow(rtemplates[k,i],interpolation='nearest',vmin=0,vmax=1)
			if(offset==0):
				if(x.shape[-1]==1):
					title(r'$c='+str(i)+'$',fontsize=20)
				else:
					title(cifarclass[i],fontsize=20)
	                xticks([])
	                yticks([])
	else:
		subplot(1,4,2+offset)
                if(x.shape[-1]==1 or len(Ax1)>55000):
                        imshow(rtemplates[k,y[k],:,:,0],cmap='gray',interpolation='nearest',vmin=0,vmax=1)
                else:
                        imshow(rtemplates[k,y[k]],interpolation='nearest',vmin=0,vmax=1)
                if(offset==0):
                        if(x.shape[-1]==1):
                                q.append(y[k])
                                title(r'$c='+str(y[k])+'$',fontsize=20)
                        else:
                                q.append(cifarclass[y[k]])
                                title(cifarclass[y[k]],fontsize=20)
                xticks([])
                yticks([])
		z.append(templates[k,y[k]])
		otherindexes=find(asarray(range(10))!=y[k])[randint(0,9,2)]
                for i,kk in zip(otherindexes,range(2)):
			z.append(templates[k,i])
                        subplot(1,4,3+offset+kk)
                        if(x.shape[-1]==1 or len(Ax1)>55000):
				q.append(i)
                                imshow(rtemplates[k,i,:,:,0],cmap='gray',interpolation='nearest',vmin=0,vmax=1)
                        else:
                                q.append(cifarclass[i])
                                imshow(rtemplates[k,i],interpolation='nearest',vmin=0,vmax=1)
                        if(offset==0):
                                if(x.shape[-1]==1):
                                        title(r'$c='+str(i)+'$',fontsize=20)
                                else:
                                        title(cifarclass[i],fontsize=20)
                        xticks([])
                        yticks([])
		d['data']=asarray(z)
		d['label']=asarray(q)
		return d


def dohist(values,labels):
	K=(values[range(len(values)),labels].reshape((-1,1))-values).flatten()
	hist(K[K!=0],100,alpha=0.8)


def dohist2(values,labels):
        K=(values[range(len(values)),labels].reshape((-1,1))-values).flatten()
        hist(K[K!=0],100,alpha=0.8)

def extract(a,i):
	return array([a[j] for j in xrange(len(a)) if i!=j])


def cosine_similarity(a,b):
	return (a*b).sum()/(sqrt((a**2).sum()+0.0000000001)+sqrt((b**2).sum()+0.0000000001))

def extract2(a,i):
        c=array([a[j] for j in xrange(len(a)) if i!=j])
	return array([cosine_similarity(k,a[i]) for k in c])




def plot_files(DATASET,Q,l1):
	As1=[]
	As2=[]
	bs1=[]
	bs2=[]
	files = glob.glob('/mnt/project2/rb42Data/ICML_TEMPLATE/'+DATASET+'bn0_b0_l'+str(l1)+'*Q'+str(Q)+'*templates*pkl')
	print files,'/mnt/project2/rb42Data/ICML_TEMPLATE/'+DATASET+'bn0_b0_l'+str(l1)+'*Q'+str(Q)+'*templates*pkl'
	x00,y00,templates00,predsy100,preds100,Ax100,bx100,predsy200,preds200,Ax200,bx200=loadfile(files[0])
        files = glob.glob('/mnt/project2/rb42Data/ICML_TEMPLATE/'+DATASET+'bn0_b1_l'+str(l1)+'*Q'+str(Q)+'*templates*pkl')
        x01,y01,templates01,predsy101,preds101,Ax101,bx101,predsy201,preds201,Ax201,bx201=loadfile(files[0])
        files = glob.glob('/mnt/project2/rb42Data/ICML_TEMPLATE/'+DATASET+'bn1_b0_l'+str(l1)+'*Q'+str(Q)+'*templates*pkl')
        x10,y10,templates10,predsy110,preds110,Ax110,bx110,predsy210,preds210,Ax210,bx210=loadfile(files[0])
        files = glob.glob('/mnt/project2/rb42Data/ICML_TEMPLATE/'+DATASET+'bn1_b1_l'+str(l1)+'*Q'+str(Q)+'*templates*pkl')
        x11,y11,templates11,predsy111,preds111,Ax111,bx111,predsy211,preds211,Ax211,bx211=loadfile(files[0])

        print tabulate(asarray([[mean((Ax100+bx100).argmax(1)==predsy100)*100,mean((Ax200+bx200).argmax(1)==predsy200)*100,mean(Ax100.argmax(1)==predsy100)*100,mean(Ax200.argmax(1)==predsy200)*100],
		[mean((Ax101+bx101).argmax(1)==predsy101)*100,mean((Ax201+bx201).argmax(1)==predsy201)*100,mean(Ax101.argmax(1)==predsy101)*100,mean(Ax201.argmax(1)==predsy201)*100],
		[mean((Ax110+bx110).argmax(1)==predsy110)*100,mean((Ax210+bx210).argmax(1)==predsy210)*100,mean(Ax110.argmax(1)==predsy110)*100,mean(Ax210.argmax(1)==predsy210)*100],
		[mean((Ax111+bx111).argmax(1)==predsy111)*100,mean((Ax211+bx211).argmax(1)==predsy211)*100,mean(Ax111.argmax(1)==predsy111)*100,mean(Ax211.argmax(1)==predsy211)*100]]), tablefmt="latex", floatfmt=".2f")

	for i in xrange(0):
		dd=dict()
		figure(figsize=(20,7))
		subplots(x00,y00,templates00,predsy100,preds100,Ax100,bx100,predsy200,preds200,Ax200,bx200,0,i+20,0)
                subplots(x01,y01,templates01,predsy101,preds101,Ax101,bx101,predsy201,preds201,Ax201,bx201,11,i+20,0)
                subplots(x10,y10,templates10,predsy110,preds110,Ax100,bx110,predsy210,preds210,Ax210,bx210,22,i+20,0)
                subplots(x11,y11,templates11,predsy111,preds111,Ax111,bx111,predsy211,preds211,Ax211,bx211,33,i+20,0)
		tight_layout()
		savefig(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_templates'+str(i)+'.png')
#		dd['image']=x00[i+20]
#		dd['templates00']=templates00[i+20]
#                dd['templates01']=templates01[i+20]
#                dd['templates10']=templates10[i+20]
#                dd['templates11']=templates11[i+20]
#		savemat(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_templates'+str(i)+'.mat',dd)
		close()
                figure(figsize=(9,4))
#                d00=subplots(x00,y00,templates00,predsy100,preds100,Ax100,bx100,predsy200,preds200,Ax200,bx200,0,i+20,1)
                d01=subplots(x01,y01,templates01,predsy101,preds101,Ax101,bx101,predsy201,preds201,Ax201,bx201,0,i+20,1)
#                d10=subplots(x10,y10,templates10,predsy110,preds110,Ax100,bx110,predsy210,preds210,Ax210,bx210,6,i+20,1)
#                d11=subplots(x11,y11,templates11,predsy111,preds111,Ax111,bx111,predsy211,preds211,Ax211,bx211,9,i+20,1)
#		savemat(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_templates00'+str(i)+'.mat',d00)
#                savemat(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_croppedtemplates01'+str(i)+'.mat',d01)
#                savemat(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_templates10'+str(i)+'.mat',d10)
#                savemat(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_templates11'+str(i)+'.mat',d11)
                tight_layout()
                savefig(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_croppedtemplates'+str(i)+'.png')
		close()
	figure(figsize=(17,5))
	subplot(1,4,1)
       	dohist(Ax100,predsy100)
	axvline(x=0,color='k',linewidth=2)
	title('No BN, No Bias',fontsize=24)
	subplot(1,4,2)
        dohist(Ax101,predsy101)
	axvline(x=0,color='k',linewidth=2)
	title('No BN, Bias',fontsize=24)
        subplot(1,4,3)
        dohist(Ax110,predsy110)
	axvline(x=0,color='k',linewidth=2)
	title('BN, No Bias',fontsize=24)
        subplot(1,4,4)
        dohist(Ax111,predsy111)
	axvline(x=0,color='k',linewidth=2)
	title('BN, Bias',fontsize=24)
	tight_layout()
	savefig(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_templateshistogram.png')
	close()
####################################################################################################################################################################
        figure(figsize=(17,5))
        subplot(1,4,1)
	vn = concatenate([extract(Ax100[i],predsy100[i]) for i in xrange(len(Ax100))])
        vp = Ax100[range(len(Ax100)),predsy100]
        hist(vp,color='g',bins=200,alpha=0.5)
        hist(vn,color='r',bins=200,alpha=0.5)
        axvline(x=0,color='k',linewidth=2)
        title('No BN, No Bias',fontsize=24)
        subplot(1,4,2)
        vn = concatenate([extract(Ax101[i],predsy101[i]) for i in xrange(len(Ax101))])
        vp = Ax101[range(len(Ax101)),predsy101]
        hist(vp,color='g',bins=200,alpha=0.5)
        hist(vn,color='r',bins=200,alpha=0.5)
        axvline(x=0,color='k',linewidth=2)
        title('No BN, Bias',fontsize=24)
        subplot(1,4,3)
        vn = concatenate([extract(Ax110[i],predsy110[i]) for i in xrange(len(Ax110))])
        vp = Ax110[range(len(Ax110)),predsy110]
        hist(vp,color='g',bins=200,alpha=0.5)
        hist(vn,color='r',bins=200,alpha=0.5)
        axvline(x=0,color='k',linewidth=2)
        title('BN, No Bias',fontsize=24)
        subplot(1,4,4)
        vn = concatenate([extract(Ax111[i],predsy111[i]) for i in xrange(len(Ax111))])
        vp = Ax111[range(len(Ax111)),predsy111]
        hist(vp,color='g',bins=200,alpha=0.5)
        hist(vn,color='r',bins=200,alpha=0.5)
        axvline(x=0,color='k',linewidth=2)
        title('BN, Bias',fontsize=24)
        tight_layout()
        savefig(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_bimodaltemplateshistogram.png')
        close()
############################################################################################################################################################################
	figure(figsize=(5,5))
	subplot(1,1,1)
        dohist(Ax101,predsy101)
	axvline(x=0,color='k',linewidth=2)
	tight_layout()
	savefig(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_01croppedtemplateshistogram.png')
	close()
##########################
        figure(figsize=(5,5))
        subplot(1,1,1)
        vn = concatenate([extract(Ax101[i],predsy101[i]) for i in xrange(len(Ax101))])
        vp = Ax101[range(len(Ax101)),predsy101]
        hist(vp,color='g',bins=200,alpha=0.5)
        hist(vn,color='r',bins=200,alpha=0.5)
        axvline(x=0,color='k',linewidth=2)
        tight_layout()
        savefig(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_01croppedbimodaltemplateshistogram.png')
        close()
##########################
        figure(figsize=(5,5))
        subplot(1,1,1)
        dohist(Ax111,predsy111)
        axvline(x=0,color='k',linewidth=2)
        tight_layout()
        savefig(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_11croppedtemplateshistogram.png')
        close()
##########################
        figure(figsize=(5,5))
        subplot(1,1,1)
        vn = concatenate([extract(Ax111[i],predsy111[i]) for i in xrange(len(Ax111))])
        vp = Ax111[range(len(Ax111)),predsy111]
        hist(vp,color='g',bins=200,alpha=0.5)
        hist(vn,color='r',bins=200,alpha=0.5)
        axvline(x=0,color='k',linewidth=2)
        tight_layout()
        savefig(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_11croppedbimodaltemplateshistogram.png')
        close()








#print 'MNIST SMALLCNN L0'
#plot_files('MNIST*smallCNN',Q=0,l1=0)
#print 'MNIST SMALLCNN L1'
#plot_files('MNIST*smallCNN',Q=0,l1=1)
print 'MNIST LARGECNN L0'
plot_files('MNIST*largeCNN',Q=0,l1=0)
#print 'MNIST LARGECNN L1'
#plot_files('MNIST*largeCNN',Q=0,l1=1)

print 'CIFAR SMALLCNN L0'
#plot_files('CIFAR*smallCNN',Q=0,l1=0)
print 'CIFAR SMALLCNN L1'
#plot_files('CIFAR*smallCNN',Q=0,l1=1)
#plot_files('CIFAR*smallRESNET',Q=0,l1=0)
print 'CIFAR LARGECNN L1'
#plot_files('CIFAR*largeCNN',Q=0,l1=1)



##
