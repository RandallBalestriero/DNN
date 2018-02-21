from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

lr      = 0.00001

if(int(sys.argv[-1])==0):
	m = smallCNN
	m_name = 'smallCNN'
elif(int(sys.argv[-1])==1):
	m = NNlargeCNN
	m_name = 'NNlargeCNN'

elif(int(sys.argv[-1])==2):
        m = resnet_small
        m_name = 'resnetSmall'



for DATASET in ['MNIST','CIFAR','SVHN']:
	for bn in [0,1]:
		x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)
		n_epochs=55
		name    = DATASET+'_'+m_name+'_bn'+str(bn)+'representations.pkl'
		model1  = DNNClassifier(input_shape,m(bn=bn,n_classes=c,bias=1),lr=lr,gpu=0,Q=0,l1=0.)
		train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=n_epochs)
		p=permutation(x_train.shape[0])[:30000]
#	        p2=permutation(x_test.shape[0])[:2000]
		masks1           = model1.get_masks(x_train[p])
#		templates1       = model1.get_templates(x_train[p])
#	        masks2           = model1.get_masks(x_test[p2])
#	#	representations1 = model1.get_representations(x_train[p])
#	#	representations2 = model1.get_representations(x_test)
		f = open('/mnt/project2/rb42Data/ICML_TEMPLATE/'+name,'wb')
		cPickle.dump([x_train[p],y_train[p],masks1],f)
		f.close()




