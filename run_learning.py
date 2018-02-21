from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

lr      = 0.001

if(int(sys.argv[-1])==0):
	m = smallCNN
	m_name = 'smallCNN'
elif(int(sys.argv[-1])==1):
	m = NNlargeCNN
	m_name = 'NNlargeCNN'

elif(int(sys.argv[-1])==2):
        m = resnet_small
        m_name = 'resnetSmall'



for DATASET in ['MNIST','CIFAR']:
	for bn in [0,1]:
		x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)
		n_epochs=55
		name    = DATASET+'_'+m_name+'_bn'+str(bn)+'learning.pkl'
		model1  = DNNClassifier(input_shape,m(bn=bn,n_classes=c,bias=1),lr=lr,gpu=0,Q=0,l1=0.)
		masks = []
                p=permutation(x_train.shape[0])[:30000]
		masks.append(model1.get_masks(x_train[p]))
#		train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=1)
#                masks.append(model1.get_masks(x_train[p]))
#                train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=1)
#                masks.append(model1.get_masks(x_train[p]))
                train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=50)
                masks.append(model1.get_masks(x_train[p]))
		f = open('/mnt/project2/rb42Data/ICML_TEMPLATE/'+name,'wb')
		cPickle.dump([x_train[p],y_train[p],masks],f)
		f.close()




