from pylab import *
import sys
import tensorflow as tf
import matplotlib as mpl
#mpl.rc('text', usetex=True)
#mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
#mpl.rcParams.update(pgf_with_rc_fonts)
label_size = 25
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

from scipy.signal import convolve2d
tf.logging.set_verbosity(tf.logging.ERROR)
from sklearn.datasets import make_moons,make_circles


def count_it(masks):
        m=masks.reshape((-1,Ns[0]))
        Z=(m*(2**arange(Ns[0])[::-1].reshape((1,-1)))).sum(1)
	keys = arange(2**Ns[0])
	d=dict(zip(keys,[0]*len(keys)))
	for z in Z:
		d[z]+=1
	return sort(d.values())[::-1]


def maxpool(X,s):
	V = randn(X.shape[0]/s,X.shape[1]/s)
	for i in xrange(V.shape[0]):
		for j in xrange(V.shape[1]):
			V[i,j]=X[i*s:(i+1)*s,j*s:(j+1)*s].max()
	return V


def doit(masks,iiii):
	m=masks.reshape((-1,Ns[iiii]))
	Z=(m*(2**arange(Ns[iiii])[::-1].reshape((1,-1)))).sum(1)
	Z=Z.reshape(xx.shape)
	if(1):
		Z=abs(Z[1:,1:]-Z[1:,:-1])+abs(Z[1:,1:]-Z[:-1,1:])+abs(Z[1:,:-1]-Z[:-1,1:])+abs(Z[:-1,1:]-Z[1:,:-1])
		Z=(abs(Z)>1).astype('float32')
		Z=(convolve2d(Z,ones((4,4)),'same')>1).astype('float32')
	        contourf(xx[1:,1:], yy[1:,1:], 1-Z, alpha=0.95,cmap='gray',interpolation='nearest')
	else:
		Z=Z[1:,1:]
		contourf(xx[1:,1:], yy[1:,1:], 1-Z, alpha=0.25)
	



def doitagain(masks):
        m=masks.reshape((-1,Ns[1]))
        Z=(m*(2**arange(Ns[1])[::-1].reshape((1,-1)))).sum(1)
        Z=Z.reshape(xx.shape)
        if(1):
                Z=abs(Z[1:,1:]-Z[1:,:-1])+abs(Z[1:,1:]-Z[:-1,1:])+abs(Z[1:,:-1]-Z[:-1,1:])+abs(Z[:-1,1:]-Z[1:,:-1])
                Z=(abs(Z)>1).astype('float32')
                Z=(convolve2d(Z,ones((4,4)),'same')>1).astype('float32')
                contourf(xx[1:,1:], yy[1:,1:], 1-Z, alpha=0.85,cmap='gray',interpolation='nearest')
        else:
                Z=Z[1:,1:]
                contourf(xx[1:,1:], yy[1:,1:], 1-Z, alpha=0.25)




def doit1(masks,iiii):
        m=masks.reshape((-1,Ns[1]))
        Z=m[:,iiii]#(m*(2**arange(Ns[iiii])[::-1].reshape((1,-1)))).sum(1)
        Z=Z.reshape(xx.shape)
        if(1):
                contourf(xx, yy,Z,100, alpha=0.45,cmap='jet')
		colorbar()
        else:
                Z=Z[1:,1:]
                contourf(xx[1:,1:], yy[1:,1:], 1-Z, alpha=0.25)






execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

lr      = 0.0010515051


X,y   = make_moons(10000,noise=0.035,random_state=20)
x_,y_ = make_circles(10000,noise=0.02,random_state=20)
x_[:,1]+= 2.
y_   += 2
X     = concatenate([X,x_],axis=0)
y     = concatenate([y,y_])

X    -= X.mean(0,keepdims=True)
X    /= X.max(0,keepdims=True)
#y-=1


print y
X=X.astype('float32')
y=y.astype('int32')
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.7,stratify=y,random_state=20)
c=4
n_epochs=100
bn=0
Ns=[int(sys.argv[-2]),int(sys.argv[-1]),10]
bias=1
input_shape = (1000,2)
print shape(x_train),shape(y_train),shape(x_test),shape(y_test)



h=500
x_min, x_max = X[:, 0].min() - .15, X[:, 0].max() + .15
y_min, y_max = X[:, 1].min() - .15, X[:, 1].max() + .15
xx, yy = np.meshgrid(linspace(x_min, x_max, h),linspace(y_min, y_max, h))
xxx=xx.flatten()
yyy=yy.flatten()
DD = asarray([xxx,yyy]).astype('float32').T


def doit2():
	masks1           = model1.get_masks(DD)
	plot(x_train[find(y_train==0)[::4],0],x_train[find(y_train==0)[::4],1],'bx',alpha=0.85)
	plot(x_train[find(y_train==1)[::4],0],x_train[find(y_train==1)[::4],1],'cx',alpha=0.85)
	plot(x_train[find(y_train==2)[::4],0],x_train[find(y_train==2)[::4],1],'gx',alpha=0.85)
	plot(x_train[find(y_train==3)[::4],0],x_train[find(y_train==3)[::4],1],'rx',alpha=0.85)
	doit(masks1[0],0)
	xlim(xx.min(), xx.max())
	ylim(yy.min(), yy.max())
	xticks([])
	yticks([])


def doit22():
        masks1           = model1.get_masks(DD)
        plot(x_train[find(y_train==0)[::4],0],x_train[find(y_train==0)[::4],1],'bx',alpha=0.85)
        plot(x_train[find(y_train==1)[::4],0],x_train[find(y_train==1)[::4],1],'cx',alpha=0.85)
        plot(x_train[find(y_train==2)[::4],0],x_train[find(y_train==2)[::4],1],'gx',alpha=0.85)
        plot(x_train[find(y_train==3)[::4],0],x_train[find(y_train==3)[::4],1],'rx',alpha=0.85)
        doit(masks1[1],1)
        xlim(xx.min(), xx.max())
        ylim(yy.min(), yy.max())
        xticks([])
        yticks([])

def onehot(i,n):
	a=zeros(n)
	a[i]=1
	return a


def doit222():
        masks1           = model1.get_feature_maps(DD)
        masks2           = model1.get_masks(DD)
	figure(figsize=(Ns[1]*10,10))
	for i in xrange(Ns[1]):
	        subplot(1,Ns[1],i+1)
		doitagain(masks2[1]*onehot(i,Ns[1]).reshape((1,-1)))
		doit1(masks1[1],i)	
	        plot(x_train[find(y_train==0)[::4],0],x_train[find(y_train==0)[::4],1],'bx',alpha=0.40)
	        plot(x_train[find(y_train==1)[::4],0],x_train[find(y_train==1)[::4],1],'cx',alpha=0.40)
	        plot(x_train[find(y_train==2)[::4],0],x_train[find(y_train==2)[::4],1],'gx',alpha=0.40)
	        plot(x_train[find(y_train==3)[::4],0],x_train[find(y_train==3)[::4],1],'rx',alpha=0.40)
        	xlim(xx.min(), xx.max())
        	ylim(yy.min(), yy.max())
        	xticks([])
        	yticks([])
		title('Unit '+str(i+1),fontsize=fs)



bn     = 1
if(sys.argv[-3]=='relu'):
	model1 = DNNClassifier(input_shape,densedouble(bn=bn,n_classes=c,bias=bias,Ns=Ns,nonlinearity=tf.nn.relu),lr=lr,gpu=0,Q=0,l1=0.)
elif(sys.argv[-3]=='lrelu'):
        model1 = DNNClassifier(input_shape,densedouble(bn=bn,n_classes=c,bias=bias,Ns=Ns,nonlinearity=tf.nn.leaky_relu),lr=lr,gpu=0,Q=0,l1=0.)
elif(sys.argv[-3]=='abs'):
        model1 = DNNClassifier(input_shape,densedouble(bn=bn,n_classes=c,bias=bias,Ns=Ns,nonlinearity=tf.abs),lr=lr,gpu=0,Q=0,l1=0.)



fs=48

figure(figsize=(35,20))

subplot(241)
doit2()
title('Init.',fontsize=fs)
subplot(245)
doit22()
train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=40)
subplot(242)
doit2()
title('Epoch 40',fontsize=fs)
subplot(246)
doit22()
train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=40)
subplot(243)
doit2()
title('Epoch 80',fontsize=fs)
subplot(247)
doit22()
train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=40)
subplot(244)
doit2()
title('Epoch 120',fontsize=fs)
subplot(248)
doit22()

tight_layout()
if(sys.argv[-3]=='relu'):
	savefig('relu_double_partitioning'+str(Ns[0])+'_'+str(Ns[1])+'.png')
elif(sys.argv[-3]=='lrelu'):
        savefig('lrelu_double_partitioning'+str(Ns[0])+'_'+str(Ns[1])+'.png')
elif(sys.argv[-3]=='abs'):
        savefig('abs_double_partitioning'+str(Ns[0])+'_'+str(Ns[1])+'.png')


doit222()
tight_layout()
if(sys.argv[-3]=='relu'):
        savefig('relu_in_double_partitioning'+str(Ns[0])+'_'+str(Ns[1])+'.png')
elif(sys.argv[-3]=='lrelu'):
        savefig('lrelu_in_double_partitioning'+str(Ns[0])+'_'+str(Ns[1])+'.png')
elif(sys.argv[-3]=='abs'):
        savefig('abs_in_double_partitioning'+str(Ns[0])+'_'+str(Ns[1])+'.png')

























