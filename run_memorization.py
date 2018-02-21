from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

lr      = 0.01

if(int(sys.argv[-1])==0):
	m = denseCNN
	m_name = 'DenseCNN'
elif(int(sys.argv[-1])==1):
	m = NNlargeCNN
	m_name = 'NNlargeCNN'

elif(int(sys.argv[-1])==2):
        m = resnet_small
        m_name = 'resnetSmall'

import copy

for DATASET in ['MNIST','CIFAR','SVHN']:
        x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)
	true_y_train=copy.copy(y_train)
	y_train = y_train[permutation(len(y_train))]
	for bn,bias in zip([0,0,0],[1,1,0]):
		n_epochs=55
		name    = DATASET+'_'+m_name+'_bn'+str(bn)+'b_'+str(bias)+'_memorization.pkl'
		model1  = DNNClassifier(input_shape,m(bn=bn,n_classes=c,bias=bias),lr=lr,optimizer= Momentum,gpu=0,Q=0,l1=0.)
		train_loss,train_accu,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=n_epochs,return_train_accu=1)
		p=permutation(x_train.shape[0])[:30000]
#	        p2=permutation(x_test.shape[0])[:2000]
		masks1           = model1.get_masks(x_train[p])
#		templates1       = model1.get_templates(x_train[p])
#	        masks2           = model1.get_masks(x_test[p2])
#	#	representations1 = model1.get_representations(x_train[p])
#	#	representations2 = model1.get_representations(x_test)
		f = open('/mnt/project2/rb42Data/ICML_TEMPLATE/'+name,'wb')
		cPickle.dump([train_loss,test_loss,x_train[p],true_y_train[p],y_train[p],masks1],f)
		f.close()




