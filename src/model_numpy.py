import numpy as np

class MLP:
    def __init__(self, layer_sizes, activation="sigmoid"):
        self.L = len(layer_sizes) - 1   
        self.activation = activation
        
        self.W = [None]
        self.b = [None]

        for l in range(1, self.L+1):
            #Wl = np.random.randn(layer_sizes[l-1], layer_sizes[l]) * 0.01
            #bl = np.zeros((1, layer_sizes[l]))
            n_in = layer_sizes[l - 1]
            n_out = layer_sizes[l]
            #xavier for for softmax on the output layer
            if l == self.L:
                limit = np.sqrt(6) / np.sqrt(n_in + n_out)
                Wl = np.random.uniform(-limit, limit, size=(n_in, n_out)).astype(np.float32) 
            else:
                if self.activation in ["sigmoid"]:
                    #trying xavier init because such bad results on sig
                    limit = np.sqrt(6) / np.sqrt(n_in + n_out)
                    Wl = np.random.uniform(-limit, limit, size=(n_in, n_out)).astype(np.float32)
                elif self.activation == "relu":
                    # He / Kaiming normal
                    std = np.sqrt(2 / n_in)
                    Wl = (np.random.randn(n_in, n_out) * std).astype(np.float32)
                    
            bl = np.zeros((1, n_out), dtype=np.float32)
            self.W.append(Wl)
            self.b.append(bl)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s*(1-s)

    def relu(self, z):
        return np.maximum(0,z)

    def relu_derivative(self, z):
        return (z>0).astype(float)

    def g(self, z):
        return self.sigmoid(z) if self.activation=="sigmoid" else self.relu(z)

    def gprime(self, z):
        return self.sigmoid_derivative(z) if self.activation=="sigmoid" else self.relu_derivative(z)

    def softmax(self, z):
        expz = np.exp(z - np.max(z,axis=1,keepdims=True))
        return expz / np.sum(expz,axis=1,keepdims=True)

    def forward(self, X):
        self.x = [X]     
        self.a = [None]  

        for l in range(1, self.L+1):
            al = self.x[l-1] @ self.W[l] + self.b[l]
            self.a.append(al)

            if l == self.L:
                xl = self.softmax(al)   
            else:
                xl = self.g(al)

            self.x.append(xl)

        return self.x[self.L]

    def compute_loss(self, y_true):
        probs = self.x[self.L]
        m = y_true.shape[0]
        eps = 1e-9
        log_likelihood = -np.log(probs[np.arange(m), y_true] + eps)
        return np.mean(log_likelihood)

    def backward(self, y_true):
        m = y_true.shape[0]
        self.delta = [None]*(self.L+1)
        self.dW = [None]*(self.L+1)
        self.db = [None]*(self.L+1)

        deltaL = self.x[self.L].copy()
        deltaL[np.arange(m), y_true] -= 1
        deltaL /= m
        self.delta[self.L] = deltaL

        for l in range(self.L,0,-1):
            self.dW[l] = self.x[l-1].T @ self.delta[l]
            self.db[l] = np.sum(self.delta[l], axis=0, keepdims=True)

            if l > 1:
                self.delta[l-1] = (self.delta[l] @ self.W[l].T) * self.gprime(self.a[l-1])

    def step(self, alpha):
        for l in range(1, self.L+1):
            self.W[l] -= alpha * self.dW[l]
            self.b[l] -= alpha * self.db[l]
