import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size



def loadfile(name):
        f=open(name,'rb')
        data=cPickle.load(f)
        f.close()
        templates = data[0][0]
        x = data[0][1]
        x-=x.min((1,2,3),keepdims=True)
        x/=x.max((1,2,3),keepdims=True)
        templates-=templates.min((2,3,4),keepdims=True)
        templates/=templates.max((2,3,4),keepdims=True)
	test = data[0][2]
        predsy = data[1][0]
        preds  = data[1][1]
        Ax     = data[1][2]
        bx     = data[1][3]
	return x,templates,predsy,preds,Ax,bx,test


def subplots(x,templates,predsy,preds,Ax,bx,test,offset):
        subplot(4,13,1+offset)
        if(x.shape[-1]==1):
                imshow(x[0,:,:,0],cmap='gray',interpolation='nearest')
                xticks([])
                yticks([])
                title(str(test[-1]))
        for i in xrange(10):
                subplot(4,13,2+i+offset)
                if(x.shape[-1]==1):
                        imshow(templates[0,i,:,:,0],cmap='gray',interpolation='nearest')
                title(str((templates[0,i,:,:,0]*x[0,:,:,0]).sum()))
                xticks([])
                yticks([])
        subplot(4,13,12+offset)
        hist(concatenate([Ax[predsy==i,i] for i in xrange(10)]),100,alpha=0.5)
        hist(concatenate([Ax[predsy!=i,i] for i in xrange(10)]),100,alpha=0.5)
        subplot(4,13,13+offset)
        hist(concatenate([bx[predsy==i,i] for i in xrange(10)]),100,alpha=0.5)
        hist(concatenate([bx[predsy!=i,i] for i in xrange(10)]),100,alpha=0.5)



def plot_files(DATASET):
	files = glob.glob('../../SAVE/QUADRATIC/'+DATASET+'bn0_b0*templates*')
	x,templates,predsy,preds,Ax,bx,test=loadfile(files[0])
	subplots(x,templates,predsy,preds,Ax,bx,test,0)
        files = glob.glob('../../SAVE/QUADRATIC/'+DATASET+'bn0_b1*templates*')
        x,templates,predsy,preds,Ax,bx,test=loadfile(files[0])
        subplots(x,templates,predsy,preds,Ax,bx,test,13)
        files = glob.glob('../../SAVE/QUADRATIC/'+DATASET+'bn1_b0*templates*')
        x,templates,predsy,preds,Ax,bx,test=loadfile(files[0])
        subplots(x,templates,predsy,preds,Ax,bx,test,26)
        files = glob.glob('../../SAVE/QUADRATIC/'+DATASET+'bn1_b1*templates*')
        x,templates,predsy,preds,Ax,bx,test=loadfile(files[0])
        subplots(x,templates,predsy,preds,Ax,bx,test,39)
	tight_layout()
	show()


#plot_files('MNIST*largeCNN')
plot_files('MNIST*smallCNN')


