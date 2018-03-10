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
        templates = data[0]
        x = data[1]
	y = data[2]
	loss = data[3]
	return templates,x,y,loss

cifarclass = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']



def roundit(n):
	return int(n*10)/10.0


def subplots(templates,x,y,loss,k,cropped,indices=0,offset=0):
	rx=x-x.min((1,2,3),keepdims=True)
	rx/=rx.max((1,2,3),keepdims=True)
	rtemplates=templates-templates.min((1,2,3,4),keepdims=True)
	rtemplates/=rtemplates.max((1,2,3,4),keepdims=True)
        d=dict()
        z=[x[k]]
	if(cropped==0):
        	subplot(4,11,1+offset)
	else:
                subplot(2,len(indices)+1,1+offset)
        if(x.shape[-1]==1):
                imshow(rx[k,:,:,0],cmap='gray',interpolation='nearest')
		q=[y[k]]
	else:
                imshow(rx[k],interpolation='nearest')
		q=[cifarclass[y[k]]]
#	if(offset==0):
	title(str(roundit(loss[-1]*100))+' %',fontsize=20)
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
					title(str(i)+',  '+str(roundit((x[k]*templates[k,i]).sum())),fontsize=18)
				else:
					title(cifarclass[i]+',  '+str(roundit((x[k]*templates[k,i]).sum())),fontsize=18)
	                xticks([])
	                yticks([])
	else:
                for i,kk in zip(indices,range(len(indices))):
			z.append(templates[k,i])
                        subplot(2,len(indices)+1,2+kk+offset)
                        if(x.shape[-1]==1):
#				q.append(i)
                                imshow(rtemplates[k,i,:,:,0],cmap='gray',interpolation='nearest',vmin=0,vmax=1)
                        else:
#                                q.append(cifarclass[i])
                                imshow(rtemplates[k,i],interpolation='nearest',vmin=0,vmax=1)
 #                       if(offset==0):
                        if(x.shape[-1]==1):
                                title(str(i)+',  '+str(roundit((x[k]*templates[k,i]).sum())),fontsize=18)
                        else:
                                title(cifarclass[i]+',  '+str(roundit((x[k]*templates[k,i]).sum())),fontsize=18)
                        xticks([])
                        yticks([])
#		d['data']=asarray(z)
#		d['label']=asarray(q)
#		return d


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




def plot_files(DATASET):
	data=[]
	files = sort(glob.glob('/mnt/project2/rb42Data/ICML_TEMPLATE/'+DATASET+'*_ctemplates*pkl'))
	for f in files:
		print f
		data.append(loadfile(f))
		print data[-1][-1][-1]
	for i in xrange(300):
		indices = concatenate([[data[0][2][i]],find(asarray(range(10))!=data[0][2][i])[permutation(9)[:2]]])
		for ii in xrange(len(files)/2):
			figure(figsize=(9,4))
			subplots(data[ii*2][0],data[ii*2][1],data[ii*2][2],data[ii*2][3],i,1,indices,0)
                        subplots(data[ii*2+1][0],data[ii*2+1][1],data[ii*2+1][2],data[ii*2+1][3],i,1,indices,4)
                	tight_layout()
			print files[ii*2],files[ii*2+1]
                	savefig(files[ii*2].split("/")[-1][:-4]+'_croppedctemplates'+str(i)+'.png')
			close()




#print 'MNIST SMALLCNN L0'
#plot_files('MNIST*smallCNN',Q=0,l1=0)
#print 'MNIST SMALLCNN L1'
#plot_files('MNIST*smallCNN',Q=0,l1=1)
print 'MNIST LARGECNN L0'
plot_files('CIFAR')
plot_files('MNIST')
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
