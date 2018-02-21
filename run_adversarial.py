from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

DATASET = sys.argv[-1]
lr      = 0.0001


 
x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)
n_epochs=100
p=permutation(len(x_train))

def doit(DATASET,m_name,m,bn,bias,l1,Q):
	data    = dict()
	name    = DATASET+'_'+m_name+'bn'+str(bn)+'_b'+str(bias)+'_adversarial.pkl'
	model1  = DNNClassifier(input_shape,m(bn=bn,n_classes=c,bias=bias,nonlinearity=resnetnonlinearity),lr=lr,gpu=0,Q=Q,l1=l1)
	train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=n_epochs)
	data['y']         = y_train[p[:1000]]
	data['x']         = x_train[p[:1000]]
	data['x_train']   = x_train[p[:input_shape[0]]]
	data['y_train']   = y_train[p[:input_shape[0]]]
        data['predictions']= model1.predict(data['x'])
#	templates         = model1.get_templates(x_train)
	data['templates'] = model1.get_templates(data['x'])
	data['masks']     = model1.get_masks(data['x_train'])
#	adv_noise         = (2*(templates>0)-1).astype('float32') 
	data['adv_noise'] = (2*(data['templates']>0)-1).astype('float32')
	noise             = (2*(randn(1000,10,input_shape[1],input_shape[2],input_shape[3])>0)-1).astype('float32')
	data['noise']     = noise#(2*(randn(input_shape[0],10,input_shape[1],input_shape[2],input_shape[3])>0)-1).astype('float32')
	data['adv_predictions']   = []
        data['noise_predictions'] = []
        data['adv_masks']         = []
        data['noise_masks']       = []
        data['adv_templates']     = []
        data['noise_templates']   = []
	for j in xrange(10):
#		data['adv_stats'].append(model1.get_template_statistics(x_train+0.01*adv_noise[:,j]))
#                data['noise_stats'].append(model1.get_template_statistics(x_train+0.01*noise[:,j]))
		data['adv_predictions'].append(model1.predict(data['x']+0.1*data['adv_noise'][:,j]))
                data['noise_predictions'].append(model1.predict(data['x']+0.1*data['noise'][:,j]))
                data['adv_masks'].append(model1.get_masks(data['x_train']+0.1*data['adv_noise'][:50,j]))
                data['noise_masks'].append(model1.get_masks(data['x_train']+0.1*data['noise'][:50,j]))
                data['adv_templates'].append(model1.get_templates(data['x']+0.1*data['adv_noise'][:,j]))
                data['noise_templates'].append(model1.get_templates(data['x']+0.1*data['noise'][:,j]))
	data['adv_predictions']=asarray(data['adv_predictions'])
	data['noise_predictions']=asarray(data['noise_predictions'])
	data['adv_templates']=asarray(data['adv_templates'])
	data['noise_templates']=asarray(data['noise_templates'])
	f = open('/mnt/project2/rb42Data/ICML_TEMPLATE/'+name,'wb')
	cPickle.dump(data,f)
	f.close()


#for q in [0,1]:
for m,m_name in zip([smallCNN,largeCNN],['smallCNN','largeCNN']):
	for l1 in [0]:
#		doit(DATASET,m_name,m,0,0,l1,0)
#		doit(DATASET,m_name,m,0,1,l1,0)
#		doit(DATASET,m_name,m,1,0,l1,0)
		doit(DATASET,m_name,m,1,1,l1,0)




