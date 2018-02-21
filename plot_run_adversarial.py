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


cifarclass = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']



def print_accu(preds):
	print shape(preds)
        advresults  = randn(10,10)
        for i in xrange(10):
                for j in xrange(10):
                        advresults[i,j]=mean((preds[:,i,:].argmax(1)==j).astype('float32'))
	print tabulate(advresults)



def do_plot(data):
	rx=data['x_train']-data['x_train'].min((1,2,3),keepdims=True)
	rx/=rx.max((1,2,3),keepdims=True)
	rtemplates = data['templates']-data['templates'].min((1,2,3,4),keepdims=True)
	rtemplates/=rtemplates.max((1,2,3,4),keepdims=True)
#        for i in xrange(len(data['masks'])):
#                data['masks'][i]=data['masks'][i].reshape((len(data['masks'][i]),-1))
#		for j in xrange(10):
#			data['adv_masks'][j][i]=data['adv_masks'][j][i].reshape((len(data['masks'][i]),-1))
#                        data['noise_masks'][j][i]=data['noise_masks'][j][i].reshape((len(data['masks'][i]),-1))
#	compute A[x+epsilon](x+epsilon) and b[x+epsilon]
	advAx = ((data['x'][:,newaxis,:,:,:]+0.01*data['adv_noise'])[:,:,newaxis,:,:,:]*transpose(data['adv_templates'],[1,0,2,3,4,5])).sum((3,4,5))#(N,adv,class)
	advBx = transpose(data['adv_predictions'],[1,0,2])-advAx
        noiseAx = ((data['x'][:,newaxis,:,:,:]+0.01*data['noise'])[:,:,newaxis,:,:,:]*transpose(data['noise_templates'],[1,0,2,3,4,5])).sum((3,4,5))#(N,adv,class)
        noiseBx = transpose(data['noise_predictions'],[1,0,2])-noiseAx
#	now ocmpute A[x]x+b[x]
	Ax = (data['x'][:,newaxis,:,:,:]*data['templates']).sum((2,3,4))
	Bx = data['predictions']-Ax
#	now compute A[x+epsilon]x+b[x+epsilon]
#       now compute A[x](x+epsilon)+b[x]
        pred11 = ((data['x'][:,newaxis,:,:,:]+0.01*data['adv_noise'])[:,:,newaxis,:,:,:]*transpose(data['adv_templates'],[1,0,2,3,4,5])).sum((3,4,5))#(N,adv,class)
	print_accu(transpose(data['adv_predictions'],[1,0,2]))
        print_accu(transpose(data['noise_predictions'],[1,0,2]))
	print_accu(pred11)

	print mean((data['y']==1)==(data['predictions'].argmax(1)==1))
	for j in xrange(10):
		print mean((data['y']==1)==(data['adv_predictions'][j].argmax(1)==1))

	return 0
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


def plot_files(DATASET,Q,l1):
	files = glob.glob('/mnt/project2/rb42Data/ICML_TEMPLATE/'+DATASET+'bn1_*adversarial.pkl')
	print files
	data  = loadfile(files[1])
	do_plot(data)
	show()
#
#        savefig(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'_l'+str(l1)+'_Q'+str(Q)+'_croppedbimodaltemplateshistogram.png')
#        close()






plot_files('MNIST_smallCNN',Q=0,l1=0)



##
