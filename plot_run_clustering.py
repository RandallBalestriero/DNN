import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size


def pca(X):
	w,v=eig(dot(X.T,X))
	return v[:,argmax(w)]

def plot_files(DATASET):
	files = glob.glob('../../SAVE/QUADRATIC/'+DATASET+'*clustering*')
	f=open(files[0],'rb')
	data,labels,templates=cPickle.load(f)
	f.close()
	templates=templates.reshape((len(templates),-1))
	
	for i in xrange(10):
		subplot(2,10,i+1)
		imshow(templates[labels==i],aspect='auto',interpolation='nearest')
                subplot(2,10,i+11)
		plot(pca(templates[labels==i]))
	show()



plot_files('MNIST*smallCNN')
#plot_Ws(['largeCNN'],lrs = ['0.0005'],DATASET='CIFAR100')

