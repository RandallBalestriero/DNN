from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

DATASET = sys.argv[-1]
lr      = 0.0005


 
x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)
n_epochs=15
p=permutation(len(x_train))

def doit(DATASET,m_name,m,bn,bias,l1,Q):
	name = DATASET+'_'+m_name+'bn'+str(bn)+'_b'+str(bias)+'_l'+str(l1)+'_Q'+str(Q)+'_templates.pkl'
	model1  = DNNClassifier(input_shape,m(bn=bn,n_classes=c,bias=bias),lr=lr,gpu=0,Q=Q,l1=l1)
	train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=n_epochs)
	templates   = model1.get_templates(x_train[p[:500]])
	predictions1,Ax1,bx1 = model1.get_template_statistics(x_train)
        predictions2,Ax2,bx2 = model1.get_template_statistics(x_test)
	f = open('/mnt/project2/rb42Data/ICML_TEMPLATE/'+name,'wb')
	cPickle.dump([[templates,x_train[p[:500]],y_train[p[:500]]],[y_train[:len(Ax1)],predictions1,Ax1,bx1],[y_test[:len(Ax2)],predictions2,Ax2,bx2]],f)
	f.close()


#for q in [0,1]:
for m,m_name in zip([smallRESNET],['smallRESNET']):
	for l1 in [0]:
#		doit(DATASET,m_name,m,0,0,l1,0)
		doit(DATASET,m_name,m,0,1,l1,0)
#		doit(DATASET,m_name,m,1,0,l1,0)
		doit(DATASET,m_name,m,1,1,l1,0)




