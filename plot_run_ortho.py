import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

DATASET = 'SVHN'
models = ['smallCNN','largeCNN']
lr = ['0.0001','0.0005','0.001']
C= linspace(0,2,5)
C=(C*100).astype('int32').astype('float32')/100.0
print C
for m in models:
	cpt = 1
	for l in lr:
		all_train = []
        	all_test  = []
                all_test2 = []
		all_train_std = []
		all_test_std= []
		all_c     = []
		files     = glob.glob('../../SAVE/QUADRATIC/'+DATASET+'_'+m+'_lr'+l+'_run0_c*')
		for f in files:
                        c  = float(f.split('c')[-1][:4])
			current_train = []
			current_test  = []
			subfiles = glob.glob(f.replace('run0','run*'))
			for ff in subfiles:
				print ff,c
				fff = open(ff,'rb')
				train,test = cPickle.load(fff)
				fff.close()
				print find(c==C)
				current_train.append(train[find(c==C)[0]])
				current_test.append(test[find(c==C)[0]])
			train = asarray(current_train)#[:,0,:]
			test  = asarray(current_test)#[:,0,:]
			print train.shape,test.shape
			all_c.append(c)
			all_train.append(mean(train[:,-100:].mean(1)))
			all_train_std.append(std(train[:,-100:].mean(1)))
        	        all_test.append(mean(test[:,-5:].mean(1)))
                        all_test2.append(mean(test.max(1)))
                        all_test_std.append(std(test[:,-5:].mean(1)))
		all_train = asarray(all_train)
		all_test  = asarray(all_test)
                all_test2 = asarray(all_test2)
                all_train_std = asarray(all_train_std)
                all_test_std  = asarray(all_test_std)
		all_c = asarray(all_c)
		print all_test.shape
		print all_c
		ii = argsort(all_c)
		all_c = all_c[ii]
		all_train = all_train[ii]
		all_test = all_test[ii]
		all_test2 = all_test2[ii]
		subplot(2,len(lr),cpt)
		plot(all_c,all_train,'ko')
                fill_between(all_c,all_train_std+all_train,all_train-all_train_std,alpha=0.5,facecolor='gray')
		xticks([])
                if(l==lr[0]):
			ylabel(r'$\mathcal{L}_{CE}$',fontsize=23)
#		boxplot(all_train.T)
		title('Learning Rate:'+l,fontsize=20)
		subplot(2,len(lr),cpt+len(lr))
#		boxplot(all_test.T)
                plot(all_c,100*all_test2,'bo')
		plot(all_c,100*all_test,'ko')
                fill_between(all_c,100*all_test_std+100*all_test,100*all_test-100*all_test_std,alpha=0.5,facecolor='gray')
                xlabel(r'$\gamma$',fontsize=19)
		if(l==lr[0]):
                	ylabel('Test Accuracy',fontsize=21)
		cpt+=1
	suptitle(DATASET+' '+m,fontsize=18)
	show()	




