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
	return data


def extract(a,i):
	return array([a[j] for j in xrange(len(a)) if i!=j])


def cosine_similarity(a,b):
	return (a*b).sum()/(sqrt((a**2).sum())*sqrt((b**2).sum()))

def extract2(a,i):
        c=array([a[j] for j in xrange(len(a)) if i!=j])
	return array([cosine_similarity(k,a[i]) for k in c])




def plot_files(DATASET,Q,l1):
	As1=[]
	As2=[]
	bs1=[]
	bs2=[]
        files = glob.glob('/mnt/project2/rb42Data/ICML_TEMPLATE/'+DATASET+'bn0_b1_l'+str(l1)+'*Q'+str(Q)+'*cosine*pkl')
        templates01,predsy101=loadfile(files[0])
        files = glob.glob('/mnt/project2/rb42Data/ICML_TEMPLATE/'+DATASET+'bn1_b1_l'+str(l1)+'*Q'+str(Q)+'*cosine*pkl')
        templates11,predsy111=loadfile(files[0])
##########################
        figure(figsize=(5,5))
        subplot(1,1,1)
        vn = concatenate([extract2(templates01[i],predsy101[i]) for i in xrange(len(templates01))])
        hist(vn,color='r',bins=300,alpha=0.5)
        axvline(x=0,color='k',linewidth=2)
	xlim([-1,1])
        tight_layout()
        savefig(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_01croppedcosinetemplateshistogram.png')
        close()
##########################
        figure(figsize=(5,5))
        subplot(1,1,1)
        vn = concatenate([extract2(templates11[i],predsy111[i]) for i in xrange(len(templates11))])
        hist(vn,color='r',bins=300,alpha=0.5)
        axvline(x=0,color='k',linewidth=2)
        xlim([-1,1])
        tight_layout()
        savefig(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_11croppedcosinetemplateshistogram.png')
        close()








print 'MNIST SMALLCNN L0'
plot_files('MNIST*smallCNN',Q=0,l1=0)
#print 'MNIST SMALLCNN L1'
#plot_files('MNIST*smallCNN',Q=0,l1=1)
print 'MNIST LARGECNN L0'
plot_files('MNIST*largeCNN',Q=0,l1=0)
#print 'MNIST LARGECNN L1'
#plot_files('MNIST*largeCNN',Q=0,l1=1)

print 'CIFAR SMALLCNN L0'
plot_files('CIFAR*smallCNN',Q=0,l1=0)
print 'CIFAR SMALLCNN L1'
#plot_files('CIFAR*smallCNN',Q=0,l1=1)
plot_files('CIFAR*largeCNN',Q=0,l1=0)
print 'CIFAR LARGECNN L1'
#plot_files('CIFAR*largeCNN',Q=0,l1=1)



##
