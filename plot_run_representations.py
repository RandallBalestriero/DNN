import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def pca(X):
	f=PCA(1)
	f.fit(X)
	return X.mean(0),f.explained_variance_[0]

def distance1(x,xs):
        return 1-((x==xs).reshape((len(xs),-1))).sum(1)/float32(prod(shape(xs)[1:]))

def distance2(x,xs):
        return 1-((x*xs).sum(1))/float32(sum(x))



#names = ['ReLU 1','Max-Pool 1','ReLU 2','Max-Pool 2','ReLU 3','Max-Pool 3']

names = ['ReLU 1','Max-Pool 1','ReLU 2','Max-Pool 2','ReLU 3']


def onehotback(k,N):
	z=ones(N)
	z[k]=0
	return z

def compute_accu(templates,y):
        clusters = [[] for i in xrange(len(templates))]
	accu     = []
        for i in xrange(len(templates)):
		accu.append(0)
		for j in xrange(len(y)):
			print j
			if(i%2==0):
		                distances = distance1(templates[i][j],templates[i])
			else:
	                        distances = distance2(templates[i][j],templates[i])
			best = argmax(-distances*onehotback(j,len(y)))#argsort(distances)[:16]
			accu[i]+=int32(y[j]==y[best])#int32(y[best[0]]==y[best[1]])
		print "LAYER",i,accur[i]/(1.0*len(y))


def compute_clusters(templates,j,x,y):
        clusters = [[] for i in xrange(len(templates))]
	figure(figsize=(23,4))
        for i in xrange(len(templates)):
		if(i%2==0):
	                distances = distance1(templates[i][j],templates[i])
		else:
                        distances = distance2(templates[i][j],templates[i])
		subplot(1,len(templates),i+1)
		best = argsort(distances)[:16]
		print best
		a=concatenate([x[best[0]],x[best[1]],x[best[2]],x[best[3]]],axis=1)
		b=concatenate([x[best[4]],x[best[5]],x[best[6]],x[best[7]]],axis=1)
                c=concatenate([x[best[8]],x[best[9]],x[best[10]],x[best[11]]],axis=1)
                d=concatenate([x[best[12]],x[best[13]],x[best[14]],x[best[15]]],axis=1)
		im = concatenate([a,b,c,d],axis=0)
		if(x.shape[-1]==1):
			imshow(im[:,:,0],cmap='gray',aspect='auto',interpolation='nearest')
		else:
                        imshow(im,aspect='auto',interpolation='nearest')
		plot([0,0],[0,x.shape[1]],'r',linewidth=2.5)
                plot([x.shape[1],x.shape[1]],[0,x.shape[1]],'r',linewidth=2.5)
                plot([0,x.shape[1]],[x.shape[1],x.shape[1]],'r',linewidth=2.5)
                plot([0,x.shape[1]],[0,0],'r',linewidth=2.5)
		xticks([])
		yticks([])
		title(names[i],fontsize=20)
	tight_layout()




def compute_white(y_train,representations1,y_test,representations2,W):
	clusters=asarray([pca(representations1[y_train==k])[0] for k in xrange(10)])
	newrepresentations1=clusters[y_train]
        labels=((representations2[:,newaxis,:]-clusters[newaxis,:,:])**2).mean(2).argmin(1)
	newrepresentations2=clusters[labels]
	subplot(121)
	hist(((newrepresentations1-representations1)**2).sum(1),100)
	subplot(122)
        hist(((newrepresentations1*representations1).sum(1)/(norm(newrepresentations1,2,axis=1)*norm(representations1,2,axis=1))),100)
	figure()
	print mean(dot(representations1,W).argmax(1)==y_train),mean(dot(newrepresentations1,W).argmax(1)==y_train)
        print mean(dot(representations2,W).argmax(1)==y_test),mean(dot(newrepresentations2,W).argmax(1)==y_test)
	noise = concatenate([(newrepresentations1-representations1).flatten(),(newrepresentations2-representations2).flatten()])
	hist(noise,200)
	show()
	

def compute_accuracy(x,y,x2,y2):
	print shape(y),shape(y2)
	for l in xrange(len(x)):
		for n in [1,5,20]:
			for ii in xrange(200):
				print y[((x2[l][ii]-x[l])**2).sum(1).argmin()],y2[ii]
			m=KNeighborsClassifier(weights='distance',metric="jaccard",n_jobs=-1)#"jaccard"
			m.fit(x[l],y)
			print "ACCURACY layer",l,"n=",n,m.score(x[l],y),m.score(x2[l],y2)


def plot_files(DATASET,bn):
	files = glob.glob('/mnt/project2/rb42Data/ICML_TEMPLATE/'+DATASET+'*bn'+str(bn)+'representations.pkl')
	print files
	f=open(files[0],'rb')
	x_train,y_train,maskstrain=cPickle.load(f)
	x_train-=x_train.min((1,2,3),keepdims=True)
        x_train/=x_train.max((1,2,3),keepdims=True)
	f.close()
#	maskstrain=maskstrain[:-1]
	for i in xrange(len(maskstrain)):
		maskstrain[i]=maskstrain[i].reshape((len(maskstrain[i]),-1))
#	compute_accu(maskstrain,y_train)
	for i in xrange(100):
		compute_clusters(maskstrain,20+i,x_train,y_train)
		savefig(DATASET.split('*')[0]+'_'+DATASET.split('*')[1]+'bn'+str(bn)+'_partitioning'+str(i)+'.png')
		close()
#	compute_accuracy(maskstrain,y_train,maskstest,y_test)

for n in [0,295]:
	for random in [0,1]:
		plot_files('MNIST_epochs'+str(n)+'_random'+str(random)+'*DenseCNN',1)
                plot_files('SVHN_epochs'+str(n)+'_random'+str(random)+'*DenseCNN',1)
                plot_files('CIFAR_epochs'+str(n)+'_random'+str(random)+'*DenseCNN',1)

#for bn in [1]:
#	plot_files('MNIST*smallCNN',bn)
#	plot_files('CIFAR*smallCNN',bn)
#	plot_files('SVHN*smallCNN',bn)




