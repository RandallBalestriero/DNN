execfile('lasagne_tf.py')
execfile('utils.py')
import random

def onehot(n,k):
        z=zeros(n,dtype='float32')
        z[k]=1
        return z

class DNNClassifier(object):
	def __init__(self,input_shape,model_class,lr=0.0001,optimizer = adam,n=3,Q=0,extra=0):
		tf.reset_default_graph()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.c = model_class.c
		config.log_device_placement=True
		self.session = tf.Session(config=config)
		self.batch_size = input_shape[0]
		self.lr = lr
		with tf.device('/device:GPU:'+str(n)):
			self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
			optimizer = optimizer(self.learning_rate)
        		self.x             = tf.placeholder(tf.float32, shape=input_shape,name='x')
        	        self.y_            = tf.placeholder(tf.int32, shape=[batch_size],name='y')
                        self.template_i    = tf.placeholder(tf.int32,name='teplatei')
                        self.template_j    = tf.placeholder(tf.int32,name='teplatej')
        	        self.test_phase    = tf.placeholder(tf.bool,name='phase')
        	        self.layers        = model_class.get_layers(self.x,input_shape,test=self.test_phase)
        	        self.prediction    = self.layers[-1].output
                        if(model_class.g==0):
                            self.templates     = tf.gradients(self.prediction[self.template_i,self.template_j],self.x)[0][self.template_i]
                        else: self.templates     = tf.gradients(self.prediction[self.template_i,self.template_j],self.layers[1].output)[0][self.template_i]
                        self.loss          = tf.reduce_mean(categorical_crossentropy(self.prediction,self.y_))# + 0.00001*l1_penaly()
        	        self.variables     = tf.trainable_variables()
        	        print "VARIABLES",self.variables
			if(Q>0):
                                if(extra):
                                    print  tf.get_collection('resnet_extra')
                                    extra_loss = tf.add_n([ortho_loss4(w) for w in tf.get_collection('resnet_extra')])+ortho_loss2(self.variables[-1])
                                else:
                                    extra_loss = ortho_loss2(self.variables[-1])
        	        	self.apply_updates = optimizer.apply(self.loss+Q*extra_loss,self.variables)
			else:
                                self.apply_updates = optimizer.apply(self.loss,self.variables)
        	        self.accuracy      = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.prediction,1),tf.int32), self.y_),tf.float32))
		self.session.run(tf.global_variables_initializer())
	def _fit(self,X,y,update_time=10):
		self.e+=1
		indices = [find(y==k) for k in xrange(self.c)]
        	n_train    = X.shape[0]/self.batch_size
        	train_loss = []
        	for i in xrange(n_train):
			if(self.batch_size<self.c):
				here = [random.sample(k,1) for k in indices]
				here = [here[i] for i in permutation(self.c)[:self.batch_size]]
			else:
				here = [random.sample(k,self.batch_size/self.c) for k in indices]
			here = concatenate(here)
                        self.session.run(self.apply_updates,feed_dict={self.x:X[here],self.y_:y[here],self.test_phase:True,self.learning_rate:float32(self.lr/sqrt(self.e))})
			print i,n_train
			if(i%update_time==0):
                                train_loss.append(self.session.run(self.loss,feed_dict={self.x:X[here],self.y_:y[here],self.test_phase:True}))
        	return train_loss
        def fit(self,X,y,X_test,y_test,n_epochs=5):
		train_loss = []
		test_loss  = []
		self.e = 0
                n_test  = X_test.shape[0]/self.batch_size
		for i in xrange(n_epochs):
			print "epoch",i
#			p = permutation(X.shape[0])
			train_loss.append(self._fit(X,y))
                	acc1 = 0.0
                	for j in xrange(n_test):
                	        acc1+=self.session.run(self.accuracy,feed_dict={self.x:X_test[self.batch_size*j:self.batch_size*(j+1)],
						self.y_:y_test[self.batch_size*j:self.batch_size*(j+1)],self.test_phase:False})
                	test_loss.append(acc1/n_test)
                	print test_loss[-1]
        	return concatenate(train_loss),test_loss
	def predict(self,X,i,j):
		n = X.shape[0]/self.batch_size
		preds = []
		for j in xrange(n):
                    preds.append(session.run(tf.nn.softmax(self.prediction),feed_dict={self.x:X_test[self.batch_size*j:self.batch_size*(j+1)],self.test_phase:False}))
                return concatenate(preds)
	def get_templates(self,X):
            templates = []
            for i in xrange(self.batch_size):
                for c in xrange(self.c):
                    templates.append(self.session.run(self.templates,feed_dict={self.x:X.astype('float32'),self.test_phase:True,self.self.template_i:i,self.self.template_j:c}))
            return templates






class smallCNN:
	def __init__(self,bn=1,c=10,g=0,p=0):
		self.bn = bn
                self.g = g
                self.p = p
		self.c = c
	def get_layers(self,input_variable,input_shape,test):
		layers = [InputLayer(input_shape,input_variable)]
                if(self.g):
                    layers.append(Generator(layers[-1],test=test,p=self.p))
		layers.append(Conv2DLayer(layers[-1],32,5,test=test,bn=self.bn))
                layers.append(Pool2DLayer(layers[-1],2))
                layers.append(Conv2DLayer(layers[-1],64,5,test=test,bn=self.bn))
                layers.append(Pool2DLayer(layers[-1],2))
                layers.append(DenseLayer(layers[-1],self.c,nonlinearity=lambda x:x))
		return layers


class resnet_large:
        def __init__(self,bn=1,c=10,g=0,p=0):
                self.bn = bn
                self.g = g
                self.p = p
                self.c = c
                tf.get_collection('resnet_extra')
        def get_layers(self,input_variable,input_shape,test):
                layers = [InputLayer(input_shape,input_variable)]
                if(self.g):
                    layers.append(Generator(layers[-1],test=test,p=self.p))
                layers.append(Conv2DLayer(layers[-1],32,3,test=test,bn=0,pad='same',nonlinearity= lambda x:x))
                layers.append(Block(layers[-1],64,128,1,test=test))
                layers.append(Block(layers[-1],64,128,1,test=test))
                layers.append(Block(layers[-1],64,256,2,test=test))
                layers.append(Block(layers[-1],64,256,1,test=test))
                layers.append(Block(layers[-1],128,512,2,test=test))
                layers.append(Block(layers[-1],128,512,1,test=test))
                layers.append(Block(layers[-1],256,1024,2,test=test))
                layers.append(Block(layers[-1],256,1024,1,test=test))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.c,nonlinearity=lambda x:x))
                return layers




class largeCNN:
        def __init__(self,bn=1,c=10,g=0,p=0):
                self.bn = bn
                self.p = p
                self.g = g
		self.c = c
        def get_layers(self,input_variable,input_shape,test):
                layers = [InputLayer(input_shape,input_variable)]
                if(self.g):
                    layers.append(Generator(layers[-1],test=test,p=self.p))
                layers.append(Conv2DLayer(layers[-1],96,3,pad='same',test=test,bn=self.bn))
                layers.append(Conv2DLayer(layers[-1],96,3,pad='full',test=test,bn=self.bn))
                layers.append(Conv2DLayer(layers[-1],96,3,pad='full',test=test,bn=self.bn))
                layers.append(Pool2DLayer(layers[-1],2))
                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn))
                layers.append(Conv2DLayer(layers[-1],192,3,pad='full',test=test,bn=self.bn))
                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn))
                layers.append(Pool2DLayer(layers[-1],2))
                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn))
                layers.append(Conv2DLayer(layers[-1],192,1,test=test))
                layers.append(DenseLayer(layers[-1],self.c,nonlinearity=lambda x:x))
                return layers





