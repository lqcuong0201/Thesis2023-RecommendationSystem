import numpy as np 
import Dataset as ds

class LinUCB:
    """
    LinUCB algorithm implementation
    """

    def __init__(self, alpha, context = "user"):
        """
        Parameters
        ----------
        alpha : number
            LinUCB parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        """

        self.n_features = len(ds.features[0])
        if context == "user":
            self.context = 1
        elif context == "both":
            self.context = 2
            self.n_features *= 2

        self.A = np.array([np.identity(self.n_features)] * ds.n_arms)
        self.A_inv = np.array([np.identity(self.n_features)] * ds.n_arms)
        self.b = np.zeros((ds.n_arms, self.n_features, 1))
        self.alpha = round(alpha, 1)
        self.algorithm = "LinUCB Disjoint (α=" + str(self.alpha) + ", context:" + context + ")"
    
    def choose_arm(self, t, user, pool_idx, bucket):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        bucket: number
            0: learning, 1: deploy
        """

        A_inv = self.A_inv[pool_idx]
        b = self.b[pool_idx]

        n_pool = len(pool_idx)

        user = np.array([user] * n_pool)
        #feature vector x
        if self.context == 1:
            x = user
        else:
            x = np.hstack((user, ds.features[pool_idx]))

        x = x.reshape(n_pool, self.n_features, 1)
        theta = A_inv @ b
        #Calculate UCB
        if (bucket == 0):
            p = np.transpose(theta, (0,2,1)) @ x + self.alpha*np.sqrt(np.transpose(x, (0,2,1)) @ A_inv @ x)
        elif (bucket == 1):
            p = np.transpose(theta, (0,2,1)) @ x

        return np.argmax(p)


    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]
        if self.context == 1:
            x = np.array(user)
        else:
            x = np.hstack((user, ds.features[a]))
        x = x.reshape((self.n_features,1))

        self.A[a] += x @ x.T
        self.b[a] += reward * x
        self.A_inv[a] = np.linalg.inv(self.A[a])


class LinUCB_Hybrid:
    """
    LinUCB algorithm implementation with hybrid model
    """

    def __init__(self, alpha, context = "user"):
        """
        Parameters
        ----------
        alpha : number
            LinUCB parameter
        context: string
            'user' or 'both'(item+user): what to use as a feature vector
        """

        self.n_features = len(ds.features[0])
        self.n_features_user  = len(ds.events[0][2])

        if context == "user":
            self.context = 1
        elif context == "both":
            self.context = 2
            self.n_features *= 2

        #the number of feature of Za
        self.k = self.n_features * self.n_features_user

        self.A0 = np.array(np.identity(self.k))
        self.A0_inv = np.array(np.identity(self.k))
        self.b0 = np.zeros((self.k,1))
        
        self.B = np.array([np.zeros((self.n_features,self.k))] * ds.n_arms)
        self.A = np.array([np.identity(self.n_features)] * ds.n_arms)
        self.A_inv = np.array([np.identity(self.n_features)] * ds.n_arms)
        self.b = np.array([np.zeros((self.n_features, 1))] * ds.n_arms)

        self.alpha = round(alpha, 1)
        self.algorithm = "LinUCB Hybrid (α=" + str(self.alpha) + ", context:" + context + ")"
    
    def choose_arm(self, t, user, pool_idx, bucket):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        bucket: number 
            0: learning, 1: deloy
        """
        B =  self.B[pool_idx]
        A_inv = self.A_inv[pool_idx]
        b = self.b[pool_idx]

        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)
        B_hat = self.A0_inv @ self.b0
        B_hat = np.array([B_hat] * n_pool)
        A0_inv = np.array([self.A0_inv] * n_pool)

        if self.context == 1:
            x = user
        else:
            x = np.hstack((user, ds.features[pool_idx]))

        x = x.reshape(n_pool, self.n_features, 1)
        arti = ds.features[pool_idx]
        #z: vector is outer product of user feature and article feature
        z = []
        for i in range(n_pool):
            z.append(np.outer(arti[i],user[i]))
        z = np.array(z).reshape((n_pool, self.k, 1))
        
        z_T = np.transpose(z,(0,2,1))
        B_T = np.transpose(B,(0,2,1))
        x_T = np.transpose(x, (0,2,1))
    
        theta = A_inv @ (b - B @ B_hat)

        std = (z_T @ A0_inv @ z - 2 * (z_T @ A0_inv @ B_T @ A_inv @ x) 
                 + x_T @ A_inv @ x + x_T @ A_inv @ B @ A0_inv @ B_T @ A_inv @ x )

        #Calculate UCB
        if (bucket == 0 ):
            p = z_T @ B_hat + x_T @ theta + self.alpha*np.sqrt(std)
        elif (bucket == 1):
            p = z_T @ B_hat + x_T @ theta
        return np.argmax(p)


    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]
        if self.context == 1:
            x = np.array(user)
        else:
            x = np.hstack((user, ds.features[a]))
        x = x.reshape((self.n_features,1))

        #z: vector is outer product of user feature and article feature
        z = np.outer(ds.features[a], user).reshape((-1,1))

        self.A0 += self.B[a].T @ self.A_inv[a] @ self.B[a]
        self.b0 += self.B[a].T @ self.A_inv[a] @ self.b[a]

        self.A[a] += x @ x.T
        self.B[a] += x @ z.T
        self.b[a] += reward * x

        self.A_inv[a] = np.linalg.inv(self.A[a])

        self.A0 += (z @ z.T - self.B[a].T @ self.A_inv[a] @ self.B[a])
        self.b0 += ( reward * z - self.B[a].T @ self.A_inv[a] @ self.b[a])
        
        self.A0_inv = np.linalg.inv(self.A0)
        
class ThomsonSampling:
    """
    Thompson sampling algorithm implementation
    """

    def __init__(self):

        self.algorithm = "TS"
        self.alpha = np.ones(ds.n_arms)
        self.beta = np.ones(ds.n_arms)

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        theta = np.random.beta(self.alpha[pool_idx], self.beta[pool_idx])
        return np.argmax(theta)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        a = pool_idx[displayed]
        self.alpha[a] += reward
        self.beta[a] += 1 - reward


class Ucb1:
    """
    UCB 1 algorithm implementation
    """

    def __init__(self, alpha):
        """
        Parameters
        ----------
        alpha : number 
            ucb parameter
        """

        self.alpha = round(alpha, 1)
        self.algorithm = "UCB1 (a=" + str(self.alpha) + ")"
        #initialize mean reward values for articles
        self.q = np.zeros(ds.n_arms)
        #initialize selected number of times of article
        self.n = np.ones(ds.n_arms)

    def choose_arm(self, t, user, pool_idx, bucket):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        bucket: number 
            0: learning, 1: deploy
        """
        #Calculate UCB
        if (bucket == 0):
            ucbs = self.q[pool_idx] + np.sqrt(self.alpha * np.log(t+ 1) / self.n[pool_idx])
        elif (bucket == 1):
            ucbs = self.q[pool_idx]

        return np.argmax(ucbs)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        a = pool_idx[displayed]

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]


class Egreedy:
    """
    Epsoilon greedy algorithm implementation
    """

    def __init__(self, epsilon):
        """
        Parameters
        ----------
        epsilon : number
            Egreedy parameter
        """

        self.e = round(epsilon, 1)
        self.algorithm = "Egreedy (ε=" + str(self.e) + ")"
        #initialize mean reward values for articles
        self.q = np.zeros(ds.n_arms)
        #initialize selected number of times of article
        self.n = np.zeros(ds.n_arms)

    def choose_arm(self, t, user, pool_idx, bucket):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        bucket: number
            0: learning, 1: deploy
        """

        p = np.random.rand()

        if(bucket == 0):
            if p > self.e:
                return np.argmax(self.q[pool_idx])
            else:
                return np.random.randint(low = 0, high = len(pool_idx))
        elif (bucket == 1):
            return np.argmax(self.q[pool_idx])

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        a = pool_idx[displayed]

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]


class Egreedy_Disjoint:
    """
    Epsoilon greedy algorithm implementation
    """

    def __init__(self, epsilon, context = "user"):
        """
        Parameters
        ----------
        epsilon : number
            Egreedy parameter
        """

        self.n_features = len(ds.features[0])
        if context == "user":
            self.context = 1
        elif context == "both":
            self.context = 2
            self.n_features *= 2

        self.A = np.array([np.identity(self.n_features)] * ds.n_arms)
        self.A_inv = np.array([np.identity(self.n_features)] * ds.n_arms)
        self.b = np.zeros((ds.n_arms, self.n_features, 1))
    
        self.e = round(epsilon, 1)
        self.algorithm = "Egreedy Disjoint (ε=" + str(self.e) + ")" + ", context:" + context + ")"

    def choose_arm(self, t, user, pool_idx, bucket):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        bucket: number 
            0: learning, 1:deploy
        """

        A_inv = self.A_inv[pool_idx]
        b = self.b[pool_idx]

        n_pool = len(pool_idx)

        user = np.array([user] * n_pool)
        if self.context == 1:
            x = user
        else:
            x = np.hstack((user, ds.features[pool_idx]))
        #x: feature vector
        x = x.reshape(n_pool, self.n_features, 1)

        theta = A_inv @ b
        q = np.transpose(theta, (0,2,1)) @ x

        prob = np.random.rand()
        
        if (bucket == 0):
            if prob > self.e:
                return np.argmax(q)
            else:
                return np.random.randint(low = 0, high = len(pool_idx))
        elif (bucket == 1):
            return np.argmax(q)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]
        if self.context == 1:
            x = np.array(user)
        else:
            x = np.hstack((user, ds.features[a]))
        x = x.reshape((self.n_features,1))

        self.A[a] += x @ x.T
        self.b[a] += reward * x
        self.A_inv[a] = np.linalg.inv(self.A[a])


class Egreedy_Hybrid:
    """
    Epsoilon greedy algorithm with hyrib modle implementation
    """

    def __init__(self, epsilon, context = "user"):
        """
        Parameters
        ----------
        epsilon : number
            Egreedy parameter
        """


        self.n_features = len(ds.features[0])
        self.n_features_user  = len(ds.events[0][2])

        if context == "user":
            self.context = 1
        elif context == "both":
            self.context = 2
            self.n_features *= 2

        #the number of feature of Za
        self.k = self.n_features * self.n_features_user

        self.A0 = np.array(np.identity(self.k))
        self.A0_inv = np.array(np.identity(self.k))
        self.b0 = np.zeros((self.k,1))
        
        self.B = np.array([np.zeros((self.n_features,self.k))] * ds.n_arms)
        self.A = np.array([np.identity(self.n_features)] * ds.n_arms)
        self.A_inv = np.array([np.identity(self.n_features)] * ds.n_arms)
        self.b = np.array([np.zeros((self.n_features, 1))] *  ds.n_arms)

        self.e = round(epsilon, 1)
        self.algorithm = "Egreedy Hybrid (ε=" + str(self.e) + ")" + ", context:" + context + ")"


    def choose_arm(self, t, user, pool_idx, bucket):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        bucket: number
            0: learning, 1: deploy
        """


        B =  self.B[pool_idx]
        A_inv = self.A_inv[pool_idx]
        b = self.b[pool_idx]

        n_pool = len(pool_idx)
        user = np.array([user] * n_pool)

        B_hat = self.A0_inv @ self.b0
        B_hat = np.array([B_hat] * n_pool)
        A0_inv = np.array([self.A0_inv] * n_pool)
        
        if self.context == 1:
            x = user
        else:
            x = np.hstack((user, ds.features[pool_idx]))
        x = x.reshape(n_pool, self.n_features, 1)

        arti = ds.features[pool_idx]
        #z: vector is outer product of user feature and article feature.
        z = []
        for i in range(n_pool):
            z.append(np.outer(arti[i],user[i]))
        z = np.array(z).reshape((n_pool, self.k, 1))

        z_T = np.transpose(z,(0,2,1))
        B_T = np.transpose(B,(0,2,1))
        x_T = np.transpose(x, (0,2,1))
    
        theta = A_inv @ (b - B @ B_hat)
        #Calculate mean reward for each article.  
        q = z_T @ B_hat + x_T @ theta

        p = np.random.rand()

        if(bucket == 0): 
            if p > self.e:
                return np.argmax(q)
            else:
                return np.random.randint(low = 0, high = len(pool_idx))
        elif(bucket == 1 ):
            return np.argmax(q)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """

        a = pool_idx[displayed]
        if self.context == 1:
            x = np.array(user)
        else:
            x = np.hstack((user, ds.features[a]))
        x = x.reshape((self.n_features,1))
        z = np.outer(ds.features[a], user).reshape((-1,1))
        
        self.A0 += self.B[a].T @ self.A_inv[a] @ self.B[a]
        self.b0 += self.B[a].T @ self.A_inv[a] @ self.b[a]

        self.A[a] += x @ x.T
        self.B[a] += x @ z.T
        self.b[a] += reward * x

        self.A_inv[a] = np.linalg.inv(self.A[a])

        self.A0 += (z @ z.T - self.B[a].T @ self.A_inv[a] @ self.B[a])
        self.b0 += ( reward * z - self.B[a].T @ self.A_inv[a] @ self.b[a])

        self.A0_inv = np.linalg.inv(self.A0)


class RandomPolicy:
    """
    Epsoilon greedy algorithm implementation
    """

    def __init__(self):
        """
        Parameters
        ----------
        epsilon : number
            Egreedy parameter
        """

        self.algorithm = "Random policy"
        #initialize mean reward values for articles
        self.q = np.zeros(ds.n_arms)
        #initialize selected number of times of article
        self.n = np.zeros(ds.n_arms)

    def choose_arm(self, t, user, pool_idx, bucket):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number
            number of trial
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        bucket: number
            0: learning, 1: deploy
        """

        return np.random.randint(low = 0, high = len(pool_idx))


    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        a = pool_idx[displayed]

        self.n[a] += 1
        self.q[a] += (reward - self.q[a]) / self.n[a]
