execfile('lasagne_tf.py')
execfile('utils.py')
import random

def onehot(n,k):
        z=zeros(n,dtype='float32')
        z[k]=1
        return z

class DNNClassifier(object):
	def __init__(self,input_shape,model_class,lr=0.0001,optimizer = adam,gpu=3,Q=0,l1=0,l2=0):
		tf.reset_default_graph()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.n_classes=  model_class.n_classes
		config.log_device_placement=True
		self.session = tf.Session(config=config)
		self.batch_size = input_shape[0]
		self.lr = lr
		with tf.device('/device:GPU:'+str(gpu)):
			self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
			optimizer = optimizer(self.learning_rate)
        		self.x             = tf.placeholder(tf.float32, shape=input_shape,name='x')
        	        self.y_            = tf.placeholder(tf.int32, shape=[batch_size],name='y')
        	        self.test_phase    = tf.placeholder(tf.bool,name='phase')
        	        self.layers        = model_class.get_layers(self.x,input_shape,test=self.test_phase)
        	        self.prediction    = self.layers[-1].output
                        if(model_class.augmentation==0):
                            self.templates     = tf.stack([tf.gradients(self.prediction,self.x,tf.one_hot(tf.fill([input_shape[0]],c),self.n_classes))[0] for c in xrange(self.n_classes)])
                        else:
                            self.templates     = tf.stack([tf.gradients(self.prediction,self.layers[1].output,tf.one_hot(tf.fill([input_shape[0]],c),self.n_classes))[0] for c in xrange(self.n_classes)])
                        self.crossentropy_loss = tf.reduce_mean(categorical_crossentropy(self.prediction,self.y_))
			self.loss          = self.crossentropy_loss+ l1*l1_penaly() +l2*l2_penaly()
        	        self.variables     = tf.trainable_variables()
        	        print "VARIABLES",self.variables[-1]
			if(Q>0):
                                extra_loss = ortho_loss2(self.layers[-1].W)
        	        	self.apply_updates = optimizer.apply(self.loss+Q*extra_loss,self.variables)
			else:
                                self.apply_updates = optimizer.apply(self.loss,self.variables)
        	        self.accuracy      = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.prediction,1),tf.int32), self.y_),tf.float32))
		self.session.run(tf.global_variables_initializer())
	def _fit(self,X,y,indices,update_time=10):
		self.e+=1
                if(self.e==60 or self.e==120 or self.e==160):
                    self.lr/=5
        	n_train    = X.shape[0]/self.batch_size
        	train_loss = []
        	for i in xrange(n_train):
			if(self.batch_size<self.n_classes):
				here = [random.sample(k,1) for k in indices]
				here = [here[i] for i in permutation(self.n_classes)[:self.batch_size]]
			else:
				here = [random.sample(k,self.batch_size/self.n_classes) for k in indices]
			here = concatenate(here)
                        self.session.run(self.apply_updates,feed_dict={self.x:X[here],self.y_:y[here],self.test_phase:True,self.learning_rate:float32(self.lr)})#float32(self.lr/sqrt(self.e))})
		        if(i%100 ==0):
                            print i,n_train
			if(i%update_time==0):
                                train_loss.append(self.session.run(self.loss,feed_dict={self.x:X[here],self.y_:y[here],self.test_phase:True}))
        	return train_loss
        def fit(self,X,y,X_test,y_test,n_epochs=5):
		train_loss = []
		test_loss  = []
		self.e     = 0
		W          = []
                n_test     = X_test.shape[0]/self.batch_size
                indices    = [find(y==k) for k in xrange(self.n_classes)]
		for i in xrange(n_epochs):
			print "epoch",i
			train_loss.append(self._fit(X,y,indices))
			# NOW COMPUTE TEST SET ACCURACY
                	acc1 = 0.0
                	for j in xrange(n_test):
                	        acc1+=self.session.run(self.accuracy,feed_dict={self.x:X_test[self.batch_size*j:self.batch_size*(j+1)],
						self.y_:y_test[self.batch_size*j:self.batch_size*(j+1)],self.test_phase:False})
                	test_loss.append(acc1/n_test)
			# SAVE LAST W FOR STATISTIC COMPUTATION
                        if(i==0 or i==(n_epochs-1)):
                            W.append(self.session.run(self.layers[-1].W))
                	print test_loss[-1]
        	return concatenate(train_loss),test_loss,W
	def predict(self,X):
		n = X.shape[0]/self.batch_size
		preds = []
		for j in xrange(n):
                    preds.append(session.run(self.prediction,feed_dict={self.x:X_test[self.batch_size*j:self.batch_size*(j+1)],self.test_phase:False}))
                return concatenate(preds,axis=0)
	def get_template_statistics(self,X):
                n = X.shape[0]/self.batch_size
                As = []
		bs = []
		preds = []
                for i in xrange(n):
		    t=self.session.run(self.templates,feed_dict={self.x:X[i*self.batch_size:(i+1)*self.batch_size].astype('float32'),self.test_phase:False})
		    templates = transpose(t,[1,0,2,3,4])
		    preds.append(self.session.run(self.prediction,feed_dict={self.x:X[self.batch_size*i:self.batch_size*(i+1)],self.test_phase:False}))
		    Ax  = (templates*X[i*self.batch_size:(i+1)*self.batch_size,newaxis,:,:,:]).sum((2,3,4))
		    bx  = preds[-1]-Ax
		    As.append(Ax)
		    bs.append(bx)
                return concatenate(preds,axis=0),concatenate(As,axis=0),concatenate(bs,axis=0)
	def get_templates(self,X):
        	templates = []
		n_batch = X.shape[0]/self.batch_size
        	for i in xrange(n_batch):
			t=self.session.run(self.templates,feed_dict={self.x:X[i*self.batch_size:(i+1)*self.batch_size].astype('float32'),self.test_phase:False})
			templates.append(transpose(t,[1,0,2,3,4]))
		return concatenate(templates,axis=0)
	def get_all_masks(self,X):
                templates = [[] for i in xrange(len(self.layers)-2)]
                n_batch   = X.shape[0]/self.batch_size
                for i in xrange(n_batch):
			for j in xrange(len(templates)):
                        	templates[j].append((self.session.run(self.layers[j].output,feed_dict={self.x:X[i*self.batch_size:(i+1)*self.batch_size].astype('float32'),self.test_phase:False})>0).astype('bool'))
		for j in xrange(len(templates)):
			templates[j]=concatenate(templates[j],axis=0).astype('bool')
                return templates
        def get_all_outputs(self,X):
                templates = [[] for i in xrange(len(self.layers)-2)]
                n_batch   = X.shape[0]/self.batch_size
                for i in xrange(n_batch):
                        for j in xrange(len(templates)):
                                templates[j].append(self.session.run(self.layers[j+1].output,feed_dict={self.x:X[i*self.batch_size:(i+1)*self.batch_size].astype('float32'),self.test_phase:False}))
                for j in xrange(len(templates)):
                        templates[j]=concatenate(templates[j],axis=0).astype('bool')
                return templates
        def get_clusters(self,X):
                templates = []
                n_batch   = X.shape[0]/self.batch_size
                for i in xrange(n_batch):
                        templates.append(self.session.run(self.layers[-2].output,feed_dict={self.x:X[i*self.batch_size:(i+1)*self.batch_size].astype('float32'),self.test_phase:False}))
                return concatenate(templates,axis=0)






class smallCNN:
	def __init__(self,bn=1,n_classes=10,augmentation=0,p=0,bias=1):
		self.bn = bn
                self.augmentation = augmentation
                self.p = p
		self.bias=bias
		self.n_classes = n_classes
	def get_layers(self,input_variable,input_shape,test):
		layers = [InputLayer(input_shape,input_variable)]
                if(self.augmentation):
                    layers.append(Generator(layers[-1],test=test,p=self.p))
		layers.append(Conv2DLayer(layers[-1],32,3,test=test,bn=self.bn,bias=self.bias))
                layers.append(Pool2DLayer(layers[-1],2))
                layers.append(Conv2DLayer(layers[-1],64,3,test=test,bn=self.bn,bias=self.bias))
                layers.append(Pool2DLayer(layers[-1],2))
                layers.append(Conv2DLayer(layers[-1],128,1,test=test,bn=self.bn,bias=self.bias))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,nonlinearity=lambda x:x,bias=0))
		return layers


class resnet_large:
        def __init__(self,bn=1,n_classes=10,augmentation=0,p=0,bias=1):
                self.bn = bn
                self.augmentation = augmentation
                self.p = p
		self.bias = bias
                self.n_classes = n_classes
        def get_layers(self,input_variable,input_shape,test):
                depth = 12
                k = 2
                layers = [InputLayer(input_shape,input_variable)]
                if(self.augmentation):
                    layers.append(Generator(layers[-1],test=test,p=self.p))
                layers.append(Conv2DLayer(layers[-1],16*k,3,test=test,bn=0,pad='same',nonlinearity= lambda x:x,bias=self.bias))
                for i in xrange(depth):
                    layers.append(Block(layers[-1],16*k,1,test=test,bias=self.bias))
                layers.append(Block(layers[-1],16*k*2,2,test=test,bias=self.bias))
                for i in xrange(depth-1):
                    layers.append(Block(layers[-1],16*k*2,1,test=test,bias=self.bias))
                layers.append(Block(layers[-1],16*k*4,2,test=test,bias=self.bias))
                for i in xrange(depth-1):
                    layers.append(Block(layers[-1],16*k*4,1,test=test,bias=self.bias))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,nonlinearity=lambda x:x,bias=0))
                return layers



class resnet_small:
        def __init__(self,bn=1,n_classes=10,augmentation=0,p=0,bias=1):
                self.bn = bn
                self.augmentation = augmentation
                self.p = p
		self.bias = bias
                self.n_classes = n_classes
        def get_layers(self,input_variable,input_shape,test):
                depth = 0
                k = 1
                layers = [InputLayer(input_shape,input_variable)]
                if(self.augmentation):
                    layers.append(Generator(layers[-1],test=test,p=self.p))
                layers.append(Conv2DLayer(layers[-1],16*k,3,test=test,bn=0,pad='same',nonlinearity= lambda x:x,bias=self.bias))
                for i in xrange(depth):
                    layers.append(Block(layers[-1],16*k,1,test=test,bias=self.bias))# Resnet 4-4 straightened bottleneck
                layers.append(Block(layers[-1],16*k*2,2,test=test,bias=self.bias))
                for i in xrange(depth-1):
                    layers.append(Block(layers[-1],16*k*2,1,test=test,bias=self.bias))
                layers.append(Block(layers[-1],16*k*4,2,test=test,bias=self.bias))
                for i in xrange(depth-1):
                    layers.append(Block(layers[-1],16*k*4,1,test=test,bias=self.bias))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,nonlinearity=lambda x:x,bias=0))
                return layers



class largeCNN:
        def __init__(self,bn=1,n_classes=10,augmentation=0,p=0,bias=1):
                self.bn = bn
                self.p = p
		self.bias = bias
                self.augmentation = augmentation
		self.n_classes = n_classes
        def get_layers(self,input_variable,input_shape,test):
                layers = [InputLayer(input_shape,input_variable)]
                if(self.augmentation):
                    layers.append(Generator(layers[-1],test=test,p=self.p))
                layers.append(Conv2DLayer(layers[-1],96,3,pad='same',test=test,bn=self.bn,bias=self.bias))
                layers.append(Conv2DLayer(layers[-1],96,3,pad='full',test=test,bn=self.bn,bias=self.bias))
                layers.append(Conv2DLayer(layers[-1],96,3,pad='full',test=test,bn=self.bn,bias=self.bias))
                layers.append(Pool2DLayer(layers[-1],2))
                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn,bias=self.bias))
                layers.append(Conv2DLayer(layers[-1],192,3,pad='full',test=test,bn=self.bn,bias=self.bias))
                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn,bias=self.bias))
                layers.append(Pool2DLayer(layers[-1],2))
                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn,bias=self.bias))
                layers.append(Conv2DLayer(layers[-1],192,1,test=test,bias=self.bias))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,nonlinearity=lambda x:x,bias=0))
                return layers





