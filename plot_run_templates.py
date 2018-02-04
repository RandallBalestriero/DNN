import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

#models = ['smallCNN']
#lrs = ['0.0001','0.0005','0.001']
#C= linspace(0,2,15)
#C=(C*100).astype('int32').astype('float32')/100.0

def plot_files(DATASET):
	files = glob.glob('../../SAVE/QUADRATIC/'+DATASET+'*templates*')
	f=open(files[0],'rb')
	templates=cPickle.load(f)
	f.close()
	rtemplates=asarray(templates)
	rtemplates-=rtemplates.min((2,3,4),keepdims=True)
        rtemplates/=rtemplates.max((2,3,4),keepdims=True)

	for k in xrange(1):
		for i in xrange(10):
			figure(figsize=(18,1.7))
                        subplot(1,11,1)
                        if(rtemplates.shape[-1]==1):
	                        imshow(rtemplates[i+5*k,0][:,:,0],aspect='auto',vmin=rtemplates[i+5*k,0].min(),vmax=rtemplates[i+5*k,0].max(),cmap='gray')
				xticks([])
				yticks([])
			else:
                                imshow(rtemplates[i+5*k,0],aspect='auto')
                                xticks([])
                                yticks([])
			for j in xrange(1,11):
				subplot(1,11,j+1)
				if(rtemplates.shape[-1]==1):
					imshow(rtemplates[i+5*k,j][:,:,0],aspect='auto',vmin=rtemplates[i+5*k,1:].min(),vmax=rtemplates[i+5*k,1:].max(),cmap='gray')
	                                xticks([])
	                                yticks([])
				else:
                                        imshow(rtemplates[i+5*k,j],aspect='auto')
	                                xticks([])
	                                yticks([])
				title(str((templates[i+5*k][0]*templates[i+5*k][j]).mean()))
	show()



plot_files('MNIST*resnetSmall')
#plot_Ws(['largeCNN'],lrs = ['0.0005'],DATASET='CIFAR100')

