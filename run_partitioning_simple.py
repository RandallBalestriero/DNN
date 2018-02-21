from pylab import *
import tensorflow as tf
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



def doit(masks):
	m=masks.reshape((-1,Ns[0]))
	Z=(m*(2**arange(Ns[0])[::-1].reshape((1,-1)))).sum(1)
	Z=Z.reshape(xx.shape)
	Z=Z[1:,1:]-Z[1:,:-1]+Z[1:,1:]-Z[:-1,1:]
	Z=(Z>1).astype('float32')
	contourf(xx[1:,1:], yy[1:,1:], Z, alpha=.8,cmap='gray')
	


execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

lr      = 0.001


X,y   = make_moons(5000,noise=0.011)
x_,y_ = make_circles(5000,noise=0.002)
x_   += 2
y_   += 2
X     = concatenate([X,x_],axis=0)
y     = concatenate([y,y_])

X    -= X.mean(0,keepdims=True)
X    /= X.max(0,keepdims=True)
#y-=1


print y
X=X.astype('float32')
y=y.astype('int32')
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.5,stratify=y)
c=4
n_epochs=100
bn=0
Ns=[20,1,10]
bias=1
input_shape = (200,2)
print shape(x_train),shape(y_train),shape(x_test),shape(y_test)
model1  = DNNClassifier(input_shape,densesimple(bn=bn,n_classes=c,bias=bias,Ns=Ns),lr=lr,gpu=0,Q=0,l1=0.)
#train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=n_epochs)



h=200
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(linspace(x_min, x_max, h),linspace(y_min, y_max, h))
xxx=xx.flatten()
yyy=yy.flatten()
DD = asarray([xxx,yyy]).astype('float32').T


def doit2():
	masks1           = model1.get_masks(DD)
	plot(x_train[find(y_train==0)[::8],0],x_train[find(y_train==0)[::8],1],'bx',alpha=0.5)
	plot(x_train[find(y_train==1)[::8],0],x_train[find(y_train==1)[::8],1],'rx',alpha=0.5)
	plot(x_train[find(y_train==2)[::8],0],x_train[find(y_train==2)[::8],1],'gx',alpha=0.5)
	plot(x_train[find(y_train==3)[::8],0],x_train[find(y_train==3)[::8],1],'kx',alpha=0.5)
	doit(masks1[0])
	xlim(xx.min(), xx.max())
	ylim(yy.min(), yy.max())
	xticks([])
	yticks([])


Ns     = [20,1,10]
bn     = 0
model1 = DNNClassifier(input_shape,densesimple(bn=bn,n_classes=c,bias=bias,Ns=Ns),lr=lr,gpu=0,Q=0,l1=0.)

fs=20

subplot(241)
doit2()
title('Init.',fontsize=fs)
train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=20)
subplot(242)
doit2()
title('Epoch 20',fontsize=fs)
train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=20)
subplot(243)
doit2()
title('Epoch 40',fontsize=fs)
train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=20)
subplot(244)
doit2()
title('Epoch 60',fontsize=fs)


Ns     = [20,1,10]
bn     = 1
model1 = DNNClassifier(input_shape,densesimple(bn=bn,n_classes=c,bias=bias,Ns=Ns),lr=lr,gpu=0,Q=0,l1=0.)

subplot(245)
doit2()
title('Init.',fontsize=fs)
train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=20)
subplot(246)
doit2()
title('Epoch 20',fontsize=fs)
train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=20)
subplot(247)
doit2()
title('Epoch 40',fontsize=fs)
train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=20)
subplot(248)
doit2()
title('Epoch 60',fontsize=fs)

show()

figure()
p1_init  = []
p1_final = []
p0_init  = []
p0_final = []
active1_init  = []
active1_final = []
active0_init  = []
active0_final = []
possible1_init  = []
possible1_final = []
possible0_init  = []
possible0_final = []


for i in xrange(2):
	Ns     = [20,1,10]
	bn     = 1
	model1 = DNNClassifier(input_shape,densesimple(bn=bn,n_classes=c,bias=bias,Ns=Ns),lr=lr,gpu=0,Q=0,l1=0.)
	masks1           = model1.get_masks(x_train)
	p1_init.append(count_it(masks1[0]))
	masks1           = model1.get_masks(DD)
        values=count_it(masks1[0])
	active1_init.append(len(values[values>0]))
        possible1_init.append(len(values))
	train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=100)
        masks1           = model1.get_masks(x_train)
	p1_final.append(count_it(masks1[0]))
        masks1           = model1.get_masks(DD)
        values=count_it(masks1[0])
        active1_final.append(len(values[values>0]))
        possible1_final.append(len(values))
########
        model1 = DNNClassifier(input_shape,densesimple(bn=0,n_classes=c,bias=bias,Ns=Ns),lr=lr,gpu=0,Q=0,l1=0.)
        masks1           = model1.get_masks(x_train)
        p0_init.append(count_it(masks1[0]))
        masks1           = model1.get_masks(DD)
        values=count_it(masks1[0])
        active0_init.append(len(values[values>0]))
        possible0_init.append(len(values))
        train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=100)
        masks1           = model1.get_masks(x_train)
        p0_final.append(count_it(masks1[0]))
        masks1           = model1.get_masks(DD)
        values=count_it(masks1[0])
        active0_final.append(len(values[values>0]))
        possible0_final.append(len(values))





p0_init = asarray(p0_init)
p0_final = asarray(p0_final)
p1_init = asarray(p1_init)
p1_final = asarray(p1_final)
subplot(121)
m0init = p0_init.mean(0)
m0final = p0_final.mean(0)
plot(m0init[m0init>0],'b',linewidth=3)
plot(m0final[m0final>0],'r',linewidth=3)
print "regions active in [-1,1]2 init",mean(active0_init),mean(active0_final)
print "Max Number of regions",mean(possible0_init),mean(possible0_final)
print "regions active in training set",len(m0init[m0init>0]),len(m0final[m0final>0])


subplot(122)
m0init = p1_init.mean(0)
m0final = p1_final.mean(0)
plot(m0init[m0init>0],'b',linewidth=3)
plot(m0final[m0final>0],'r',linewidth=3)

show()










