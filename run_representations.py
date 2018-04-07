from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

lr      = 0.0005

if(int(sys.argv[-1])==0):
	m = DenseCNN
	m_name = 'DenseCNN'
elif(int(sys.argv[-1])==1):
	m = NNlargeCNN
	m_name = 'NNlargeCNN'

elif(int(sys.argv[-1])==2):
        m = resnet_small
        m_name = 'resnetSmall'



for DATASET in ['MNIST','CIFAR','SVHN']:
	for bn in [1]:
                x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)
		p=permutation(x_train.shape[0])[:30000]
		ppp = permutation(x_train.shape[0])
		for training in [295,0]:
			for rrr,random_labels in zip([1,0],[y_train[ppp],y_train]):
				name    = DATASET+'_epochs'+str(training)+'_random'+str(rrr)+'_'+m_name+'_bn'+str(bn)+'representations.pkl'
				model1  = DNNClassifier(input_shape,m(bn=bn,n_classes=c,bias=1),lr=lr,gpu=0,Q=0,l1=0.)
				train_loss,test_loss,W = model1.fit(x_train,random_labels,x_test,y_test,n_epochs=training)
				masks1           = model1.get_masks(x_train[p])
#				f = open('/mnt/project2/rb42Data/ICML_TEMPLATE/'+name,'wb')
                                f = open('/mnt/project2/rb42Data/ICML_TEMPLATE/'+name,'wb')
				cPickle.dump([x_train[p],y_train[p],masks1],f)
				f.close()




