from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

DATASET = sys.argv[-1]
lr      = 0.0005


 
x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)
n_epochs=270
p=permutation(len(x_train))

def doit(DATASET,m_name,m):
	name = DATASET+'_'+m_name+'_ctemplates.pkl'
	model1  = DNNClassifier(input_shape,m,lr=lr,gpu=0,Q=0,l1=0)
	train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=n_epochs)
	templates   = model1.get_templates(x_train[p[:500]])
	f = open('/mnt/project2/rb42Data/ICML_TEMPLATE/'+name,'wb')
	cPickle.dump([templates,x_train[p[:500]],y_train[p[:500]],test_loss],f)
	f.close()


#for q in [0,1]:
for m,m_name in zip([smallRESNET(bn=1,n_classes=10,bias=1,nonlinearity=tf.nn.relu),
			smallRESNET(bn=1,n_classes=10,bias=1,nonlinearity=tf.abs),
			smallRESNET(bn=0,n_classes=10,bias=1,nonlinearity=tf.nn.relu),
			smallRESNET(bn=0,n_classes=10,bias=1,nonlinearity=tf.abs),
                        largeCNN(bn=1,n_classes=10,bias=1,nonlinearity=tf.nn.relu),
                        largeCNN(bn=1,n_classes=10,bias=1,nonlinearity=tf.abs),
                        largeCNN(bn=0,n_classes=10,bias=1,nonlinearity=tf.nn.relu),
                        largeCNN(bn=0,n_classes=10,bias=1,nonlinearity=tf.abs)],['smallRESNET_bn1_relu','smallRESNET_bn1_abs','smallRESNET_bn0_relu','smallRESNET_bn0_abs','largeCNN_bn1_relu','largeCNN_bn1_abs','largeCNN_bn0_relu','largeCNN_bn0_abs']):
	doit(DATASET,m_name,m)




