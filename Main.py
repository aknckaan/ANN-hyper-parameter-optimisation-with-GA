import csv
from threading import Thread
from multiprocessing.pool import ThreadPool
import multiprocessing
from multiprocessing import Queue
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from enum import Enum
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import time
import pickle
from sklearn.neural_network import MLPClassifier

class NN_GA():
    is_first=True
    solver = ["lbfgs", "sgd", "adam"] #1
    #alpha float 0.0001 research 2
    learning_rate = ["constant", "adaptive"] #3
    #learning_rate_init float # 4
    #power_t float when sgd  #5
    #momentum float when sgd # 6
    #nesterovs_momentum boolean #7
    # tol float, only on sgd or adam # 8
    shape=[1]
    mask=[]
    pop_size=0
    pop=[]
    pop_std=[]
    pop_scores=[]
    classification=False
    avrg_eval=10
    minimisation=False
    ga_type = ""
    trust=[]
    immunity=[]
    reporter=""

    #options = {mask_mutation:rate,float_mutation:rate,float_range:range}

    def __init__(self, pop_size=20,reporter=None,classification=False,minimisation=False,ga_type="Gen"):
        self.pop_size = pop_size
        self.minimisation=minimisation
        self.classification=classification
        self.ga_type=ga_type
        self.immunity=np.zeros(pop_size)
        self.reporter=reporter

        for i in range(0, pop_size):
            parameters = [self.shape, self.random_int(3), self.random_float(1000),
                          self.random_int(2), self.random_float(100), self.random_float(1),self.random_float(100), self.random_int(2),self.random_float(2)]

            cur_mask=self.setMask(parameters)
            self.mask.append(cur_mask)
            self.pop.append(parameters)
            self.trust=np.zeros(pop_size)
            
    def setMask(self,parameters):
        mask_power_t = [1 if parameters[1] == 1 else 0]
        mask_momentum = [1 if parameters[1] == 1 else 0]
        mask_tol = [0 if parameters[1] == 0 else 1]
        cur_mask = [0, 1, 1, 1, 1, mask_power_t, mask_momentum, 1, mask_tol]
        return cur_mask
        
    def mutate(self,ind):
        parameters = self.pop[ind]
        cur_mask=self.mask[ind]
        # solver mutate
        if np.random.rand()>0.90: #%10
            mutation=np.round(np.random.rand()*2)
            parameters[1]=mutation
            
         # alpha mutate
        if np.random.rand() > 0.90:  # %10
            mutation = self.random_float(1000)
            parameters[2] =np.abs(parameters[2]+ mutation)
            
        #learning rate mutate
        if np.random.rand()>0.90: #%10
            mutation=np.round(np.random.rand()*1)
            parameters[3]=mutation

        # learning_rate_init
        if np.random.rand() > 0.90:  # %10
            mutation = self.random_float(100)
            parameters[4] = np.abs(parameters[4] + mutation)

        # power_t
        if np.random.rand() > 0.90 and cur_mask[5]==1:  # %10
            mutation = self.random_float(100)
            parameters[5] = parameters[5] + mutation

        # momentum
        if np.random.rand() > 0.90 and cur_mask[6] == 1:  # %10
            mutation = self.random_float(100)
            parameters[6] = np.abs(parameters[6] + mutation)
            if parameters[6] >1:
                parameters[6]=1

                #nesterovs_momentum
        if np.random.rand() > 0.90:  # %10
            parameters[7] = self.random_int(2)

        # momentum
        if np.random.rand() > 0.90 and cur_mask[8] == 1:  # %10
            mutation = self.random_float(2)
            parameters[8] = parameters[8] + mutation

        # add node
        for i in range(len(parameters[0])):
            if np.random.rand() > 0.85:
                parameters[0][i]+= 1

        # delete node
        for i in range(len(parameters[0])):
            if np.random.rand() > 0.95:
                if parameters[0][i]>1:
                    parameters[0][i]-= 1

        # add layer
        if np.random.rand() > 0.97:
            parameters[0]= parameters[0] + [1]
            self.immunity[ind]=5

        self.mask[ind]=self.setMask(parameters)

    def random_int(self,range):
       return np.round(np.random.rand()*(range-1))

    def random_float(self,scale):
        return np.random.rand()* 1/scale - 1/(2*scale)

    def run_GA(self,X,Y):
        if self.ga_type=="Gen":
            res=self.pop_evaluation(X,Y)

        else:
            res=self.pop_evaluation_ss(X, Y)

        if not self.reporter==None:
            self.reporter.imunity=self.immunity
            self.reporter.networks=self.pop
            self.reporter.trust=self.trust

        return res

    def ss_evaluate(self,X,Y):

        self.trust=[i+1 for i in self.trust]
        #select good and produce offspring
        pop_scores = self.pop_scores

        if self.minimisation:
            score = np.min(pop_scores)
        else:
            score = np.max(pop_scores)

        adjusted = np.abs(pop_scores)

        if self.minimisation:
            self.trust=[10 if j>10 else j for j in self.trust]
            adjusted = [(i-(i/20)*(j+1))for i,j in zip(adjusted,self.trust)]
            adjusted = [max(adjusted) - i for i in adjusted]
        else:
            self.trust = [10 if j > 10 else j for j in self.trust]
            adjusted = [(i + (i / 20) * (j + 1)) for i, j in zip(adjusted, self.trust)]

        wheel = np.cumsum(adjusted)
        wheel = [wheel[i] - adjusted[0] for i in range(len(wheel))]

        index1 = int(np.round(np.random.rand() * np.max(wheel)))
        index1 = min(range(len(wheel)), key=lambda k: abs(wheel[k] - index1))
        index2 = int(np.round(np.random.rand() * np.max(wheel)))
        index2 = min(range(len(wheel)), key=lambda k: abs(wheel[k] - index2))
        gene1, mask1 = self.crossover(index1, index2)
        gene2, mask2 = self.crossover(index1, index2)
        score1=self.evaluate_ss(gene1,X,Y)
        score1=self.inv_score(score1)
        score2=self.evaluate_ss(gene2,X,Y)
        score2=self.inv_score(score2)

        again=True
        #select bad
        while (again):
            again = False
            pop_scores = self.pop_scores

            if not self.minimisation:
                score = np.min(pop_scores)
            else:
                score = np.max(pop_scores)

            index = np.where(pop_scores == score)[0][0]

            adjusted = np.abs(pop_scores)

            if not self.minimisation:
                adjusted = [max(adjusted) - i for i in adjusted]

            wheel = np.cumsum(adjusted)
            wheel = [wheel[i] - adjusted[0] for i in range(len(wheel))]


            index3 = int(np.round(np.random.rand() * np.max(wheel)))
            index3 = min(range(len(wheel)), key=lambda k: abs(wheel[k] - index3))
            index4 = int(np.round(np.random.rand() * np.max(wheel)))
            index4 = min(range(len(wheel)), key=lambda k: abs(wheel[k] - index4))

            if self.immunity[index3]>0:
                again = True
                self.mutate(index3)
                self.pop_scores[index3]= self.inv_score(self.evaluate_ss(self.pop[index3],X,Y))
                self.immunity[index3]-=1

            if self.immunity[index4] > 0:
                again = True
                self.mutate(index4)
                self.pop_scores[index4] = self.inv_score(self.evaluate_ss(self.pop[index4], X, Y))
                self.immunity[index4] -= 1


        pool=[self.pop[index1],self.pop[index2],gene1,gene2]
        pool_mask=[self.mask[index1],self.mask[index2],mask1,mask2]
        replacing=[self.pop_scores[index1],self.pop_scores[index2],score1,score2]

        if self.minimisation:
            replacing_index=replacing.index(min(replacing))
        else:
            replacing_index = replacing.index(max(replacing))

        self.pop[index3]=pool[replacing_index]
        self.mask[index3]=pool_mask[replacing_index]
        self.pop_scores[index3]=replacing[replacing_index]

        replacing.remove(replacing[replacing_index])
        pool.remove(pool[replacing_index])
        pool_mask.remove(pool_mask[replacing_index])

        if self.minimisation:
            replacing_index = replacing.index(min(replacing))
        else:
            replacing_index = replacing.index(max(replacing))

        self.pop[index4] = pool[replacing_index]
        self.mask[index4] = pool_mask[replacing_index]
        self.pop_scores[index4] = replacing[replacing_index]

        self.mutate(index3)
        self.mutate(index4)

        self.trust[index3] = 0
        self.trust[index4] = 0

    def pop_evaluation_ss(self,X,Y):
        if self.is_first:
            self.is_first = False
            return self.pop_evaluation(X,Y)
        else:
            self.ss_evaluate(X,Y)
            return self.pop_scores, self.pop_std

    def pop_evaluation(self,X,Y):
        data = np.zeros(self.pop_size)
        average_score = np.zeros(self.pop_size)
        stds = np.zeros(self.pop_size)
        thread_pool=[]
        start = time.time()
        q = Queue()
        for i in range(self.pop_size):
            t1 = Thread( target=self.evaluate,
                                   args=(q,i,X,Y))
            t1.start()
            #thread_pool.append(t1)


        cpu_count = multiprocessing.cpu_count()
        max_size = len(thread_pool)
        j = 0 # total
        waiting=0
        running=[]

        pool = ThreadPool(self.pop_size)
        results = pool.map(target=self.evaluate, args=(X,Y))
        pool.close()
        pool.join()
        print(results)



        end = time.time()
        print("evaluation took"),
        print(end - start)
        self.avrg_eval=(end-start)/self.pop_size
        self.pop_std=stds
        self.pop_scores=data

        start = time.time()
        self.pop,self.mask=self.selection()
        end = time.time()
        print("selection took"),
        print(end - start)

        start = time.time()
        for i in range(len(self.pop)):
            self.mutate(i)
        end = time.time()
        print("mutation took"),
        print(end - start)
        return self.pop_scores,self.pop_std

    def selection(self):
        
        new_pop=[]
        new_mask=[]
        pop_scores=self.pop_scores

        if self.minimisation:
            score = np.min(pop_scores)
        else:
            score=np.max(pop_scores)

        index=np.where(pop_scores == score)[0][0]

        pickle.dump(self.pop[index],open('fittest.p','wb'))


        new_pop.append(self.pop[index])
        new_mask.append(self.mask[index])
        adjusted=np.abs(pop_scores)

        if self.minimisation:
            adjusted=[max(adjusted)-i for i in adjusted]

        wheel=np.cumsum(adjusted)
        wheel=[wheel[i]-adjusted[0] for i in range(len(wheel))]
    
        for i in range(len(wheel)-1):
            index1=int(np.round(np.random.rand()*np.max(wheel)))
            index1= min(range(len(wheel)), key=lambda k: abs(wheel[k] - index1))
            index2=int(np.round(np.random.rand()*np.max(wheel)))
            index2 = min(range(len(wheel)), key=lambda k: abs(wheel[k] - index2))
            gene,mask=self.crossover(index1,index2)
            new_pop.append(gene)
            new_mask.append(mask)
            
        return new_pop,new_mask
            
    def crossover(self,index1,index2):
        

        ind_1=self.pop[index1]
        ind_2=self.pop[index2]
        swap = int(np.round(np.random.rand() * len(ind_1)))
        ind1_mask=self.mask[index1]
        ind2_mask = self.mask[index2]

        new_gene=ind_1[:swap]+ind_2[swap:]
        new_mask=ind1_mask[:swap]+ind2_mask[swap:]

        return new_gene,new_mask
    
    def inv_score(self,scores):
        scores=np.abs(scores)
        scores=scores.tolist()
        if min:
            scores.sort(reverse=True)
        else:
            scores.sort(reverse=False)
        #scores.sort(reverse=True)

        score=0
        for i in range(len(scores)):
            score+=scores[i]/(i+1)

        score=score*1000
        return float(score)/len(scores)

    def evaluate_ss(self, gene, X, Y):
        parameters = gene
        hidden_layers = parameters[0]
        # a,b=vars(self.solver).items()

        cur_solver = (self.solver)[int(parameters[1])]
        cur_learning_rate = (self.learning_rate)[int(parameters[3])]
        cur_nester = False
        if parameters[7] == 1:
            cur_nester = True

        cur_momentum = parameters[6]

        if self.classification:
            reg = MLPClassifier(hidden_layer_sizes=hidden_layers, activation="relu", solver=cur_solver,
                                alpha=parameters[2],
                                batch_size='auto', learning_rate=cur_learning_rate, learning_rate_init=parameters[4],
                                power_t=parameters[5],
                                max_iter=200, shuffle=True, random_state=None, tol=parameters[8], verbose=False,
                                warm_start=False,
                                momentum=cur_momentum, nesterovs_momentum=cur_nester, early_stopping=False,
                                validation_fraction=0.1,
                                beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

        else:

            reg = MLPRegressor(hidden_layer_sizes=hidden_layers, activation="relu", solver=cur_solver,
                               alpha=parameters[2],
                               batch_size='auto', learning_rate=cur_learning_rate, learning_rate_init=parameters[4],
                               power_t=parameters[5],
                               max_iter=200, shuffle=True, random_state=None, tol=parameters[8], verbose=False,
                               warm_start=False,
                               momentum=cur_momentum, nesterovs_momentum=cur_nester, early_stopping=False,
                               validation_fraction=0.1,
                               beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

        score = cross_val_score(reg, X, Y, cv=3)

        return score

    def evaluate(self,q,index,X,Y):
        parameters=self.pop[index]
        hidden_layers=parameters[0]
        #a,b=vars(self.solver).items()
            
        cur_solver =  (self.solver)[int(parameters[1])]
        cur_learning_rate = (self.learning_rate)[int(parameters[3])]
        cur_nester=False
        if parameters[7] == 1:
            cur_nester=True

        cur_momentum=parameters[6]

        if self.classification:
            reg = MLPClassifier(hidden_layer_sizes=hidden_layers, activation="relu", solver=cur_solver,
                               alpha=parameters[2],
                               batch_size='auto', learning_rate=cur_learning_rate, learning_rate_init=parameters[4],
                               power_t=parameters[5],
                               max_iter=200, shuffle=True, random_state=None, tol=parameters[8], verbose=False,
                               warm_start=False,
                               momentum=cur_momentum, nesterovs_momentum=cur_nester, early_stopping=False,
                               validation_fraction=0.1,
                               beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

        else:

            reg = MLPRegressor(hidden_layer_sizes=hidden_layers, activation="relu", solver=cur_solver, alpha=parameters[2],
                               batch_size='auto', learning_rate=cur_learning_rate, learning_rate_init=parameters[4], power_t=parameters[5],
                               max_iter=200, shuffle=True, random_state=None, tol=parameters[8], verbose=False, warm_start=False,
                               momentum=cur_momentum, nesterovs_momentum=cur_nester, early_stopping=False, validation_fraction=0.1,
                               beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

        score = cross_val_score(reg, X, Y, cv=2)

        q.put([score,index])

class Reporter():
    imunity=[]
    networks=[]
    fittest_index=[]
    trust=[]

def prepareData():
    Y=[]
    list = []
    seen = {}
    cur_num = 0
    count=-1
    with open('all/train_V2.csv', 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            count += 1
            if count==0  or count >100:
                continue
                
            if row[1] not in seen:
                seen[row[1]] = float(cur_num)
                row[1] = float(cur_num)
                cur_num += 1
            else:
                row[1] = seen[row[1]]

            if row[2] not in seen:
                seen[row[2]] = float(cur_num)
                row[2] = float(cur_num)
                cur_num += 1
            else:
                row[2] = seen[row[2]]

            if row[15] not in seen:
                seen[row[15]] = float(cur_num)
                row[15] = float(cur_num)
                cur_num += 1
            else:
                row[15] = seen[row[15]]

            try:
                Y.append(float(row[-1]))
            
                row = row[1:-1]
                row = [float(i) for i in row]
                list.append(row)
            except:
                print(row[-1])
        
        enc = OrdinalEncoder()
        X=enc.fit_transform(list,Y)
        
    return X,Y

def random_partition(X,Y,size):
    partition=np.floor(np.random.rand(size)*len(X))
    partition=[int(i) for i in partition]
    partition=np.asarray(partition)
    
    x_new=[X[i] for i in partition]
    y_new=[Y[i] for i in partition]
    
    return x_new,y_new


start = time.time()
X,Y=prepareData()
end = time.time()
print("prepare data took "),
print(end - start)

start = time.time()
X = Normalizer().fit_transform(X,Y)
end = time.time()
print("normaliser took "),
print(end - start)

#X=pickle.load(open('datax.p','rb'))
#Y=pickle.load(open('datay.p','rb'))

scores=[]
fittest=[]
std_list=[]
fittest_std=[]


mini=True
clsfc=False
gen=0
rprt=Reporter()
ga=NN_GA(30,minimisation=mini,classification=clsfc,ga_type="SS",reporter=rprt)
while True:
    fig, (ax, bx, cx, dx, ex) = plt.subplots(5, 1, sharex=False)
    print("__________________________")
    print(gen)
    start = time.time()
    if gen==0:
        partx, party = random_partition(X, Y, 1000)
    else:
        partx, party = random_partition(X, Y, 500000)

    end = time.time()
    print("partition took "),
    print(end - start)

    start = time.time()
    score,stds=ga.run_GA(partx,party)
    end = time.time()
    print("ga gen took "),
    print(end - start)
    score=score.tolist()
    if mini:
        fittest.append(min(np.abs(score)))
        ind=score.index(min(np.abs(score)))
    else:
        fittest.append(max(np.abs(score)))
        ind = score.index(max(np.abs(score)))

    fittest_std.append(stds[ind])
    scores.append(np.mean(score))
    ax.plot(fittest[-10:], label='Fittest', color='orange')
    ax.plot(scores[-10:], label='Fittest', color='blue')
    std=np.mean(stds)
    std_list.append(std)
    upper = [x + y for x, y in zip(scores[-10:], std_list[-10:])]
    lower = [x - y for x, y in zip(scores[-10:], std_list[-10:])]
    ax.fill_between(range(0, len(upper)), upper, lower, facecolor='blue', alpha=0.2)

    upper = [x + y for x, y in zip(fittest[-10:], fittest_std[-10:])]
    lower = [x - y for x, y in zip(fittest[-10:], fittest_std[-10:])]
    ax.fill_between(range(0, len(upper)), upper, lower, facecolor='orange', alpha=0.2)

    bx.bar(range(len(fittest[-10:])), fittest[-10:],color='orange')

    cx.bar(range(len(rprt.imunity)),rprt.imunity,color="blue")
    dx.bar(range(len(rprt.trust)),rprt.trust,color="blue")
    cx.bar((ind), rprt.imunity[ind], color="orange")
    dx.bar((ind), rprt.trust[ind], color="orange")

    data_plot=np.zeros(10)
    data_count=np.zeros(10)
    max_seen=0
    for i in rprt.networks:
        layers = i[0]
        if len(layers)>max_seen:
            max_seen=len(layers)

        for i in range(len(layers)):
            data_plot[i]+=layers[i]
            data_count[i]+=1

    data_plot=data_plot[:max_seen]

    for i in range(len(data_plot)):
        data_plot[i]=data_plot[i]/data_count[i]

    ex.bar(range(len(data_plot)), data_plot, color="blue")

    cx.set_xlabel("imunity",fontsize=6)
    dx.set_xlabel("trust",fontsize=6)
    ex.set_xlabel("composition",fontsize=6)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    for tick in bx.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    for tick in bx.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    for tick in dx.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    for tick in ex.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    for tick in dx.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    for tick in ex.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    for tick in cx.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    for tick in cx.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)


    fig.tight_layout()
    fig.show()
    #fig.tightLayout()

    fig.savefig('graph.png')
    plt.pause(0.01)
    print("__________________________")
    gen+=1