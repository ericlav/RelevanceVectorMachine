import numpy as np
from sklearn.base import BaseEstimator
from sklearn.gaussian_process.kernels import RBF
from functools import partial


# Definition des kernels

def linear_kernel(X1, X2):
	#Retourne la matrice de Gram d'un noyau linéaire
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = np.dot(x1.T, x2)
    return gram_matrix

def gaussian_kernel(X1, X2, sigma=1):
	#Retourne la matrice de Gram d'un noyau gaussien
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.flatten()
            x2 = x2.flatten()
            gram_matrix[i, j] = np.exp(- np.sum( np.power((x1 - x2),2) ) / float(2*(sigma**2)*X1.shape[1]) )
    return gram_matrix

def linear_spline_kernel(X1, X2):
	#Retourne la matrice de Gram d'un noyau lineaire spline
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = np.prod(1 + x1*x2 + x1*x2*np.minimum(x1, x2) \
            - 0.5*(x1+x2)*(np.minimum(x1, x2)**2) + (1/3)*(np.minimum(x1, x2)**3))
    return gram_matrix

def rbf_kernel(X1, X2):
	#Retourne la matrice de Gram du noyau RBF de sklearn
    kernel = RBF()
    return kernel.__call__(X1, X2)



# RVM

class RVM(BaseEstimator):
	#Module RVM qui reprend le format type de sklearn
	
    def __init__(self, kernel=gaussian_kernel, kernel_param=0.5, iter_max=2000, seuil_convergence=10^(-1), selection="aleat"):
		#Initialise un modèle RVM
        self.kernel_param = kernel_param
        self.kernel = kernel
        self.iter_max = iter_max
        self.iter_total = 0
        self.seuil_convergence = seuil_convergence #10^-6 dans l'article
        self.selection = selection #type de sélection des composantes utilisée dans l'algorithme

    def compute_A(self):
		# Calcule la matrice A utile pour des calculs ultérieurs
        return np.diag(self.alpha[self.index])

    def compute_sigma_and_mu(self, A, y):
		# Calcule sigma et mu les paramètres de la distribution a posteriori des poids w
		
        # Calcul sigma
        inv_sigma = self.beta * self.phi[:, self.index].T @ self.phi[:, self.index] + A
        self.sigma = np.linalg.inv(inv_sigma)

        # Calcul mu
        self.mu = self.beta * self.sigma @ self.phi[:, self.index].T @ y

    def compute_S_and_Q(self, A, y):
		#Calcule S et Q pour évaluer ensuite les composantes à rajouter/supprimer
        S = np.zeros(self.M)
        s = np.zeros(self.M)
        Q = np.zeros(self.M)
        q = np.zeros(self.M)

        C = (1/self.beta) * np.eye(self.N) + self.phi[:, self.index] @ np.linalg.inv(A) @ self.phi[:, self.index].T
        C_inv = np.linalg.inv(C)

        for m in range(self.M):
            phi_m_T_C_inv = self.phi[:,m].T @ C_inv
            S[m] = phi_m_T_C_inv @ self.phi[:,m]
            Q[m] = phi_m_T_C_inv @ y
            if m in self.index:
                s[m] = self.alpha[m]*S[m]/(self.alpha[m]-S[m])
                q[m] = self.alpha[m]*Q[m]/(self.alpha[m]-S[m])
            else:
                s[m] = S[m]
                q[m] = Q[m]

        return S, Q, s, q

    # Mise à jour des alpha_i

    def reestimate(self, i, s, q):
		#Réestime la valeur du alpha associé à la i-eme composante
        old_alpha_i = self.alpha[i]
        self.alpha[i] = s[i]**2/(q[i]**2-s[i])

    def add(self, i, s, q):
		#Ajoute la composante i au modèle
        self.index.append(i)
        self.reestimate(i, s, q)

    def delete(self, i):
		#Supprime la composante i du modèle
        self.index.remove(i)
        self.alpha[i] = np.inf

    #Reestimation de beta (sigma2)

    def recompute_beta(self, y):
		#Réestime la valeur du paramètre beta
        sigma_2 = np.linalg.norm(y - self.phi[:,self.index] @ self.mu, ord=2)**2 / (self.N - len(self.index) + (self.alpha[self.index]*np.diagonal(self.sigma)).sum())
        self.beta = 1 / sigma_2

    #Condition de convergence de l'algorithme d'apprentissage des RVM

    def check_convergence(self, theta, s, q):
		#Teste si la condition de convergance est vérifiée
		
        #Check if diff of log alpha is low for all relevant basis functions
        alpha_updated_in_index = s[self.index]**2/(q[self.index]**2-s[self.index])
        alpha_diff_in_index = np.abs(np.log(self.alpha[self.index]) - np.log(alpha_updated_in_index))
        first_cond = (alpha_diff_in_index < self.seuil_convergence).all()
        #Check if theta_i <= 0 for all basis functions not in the model
        sd_cond = (theta[[i for i in range(self.M) if i not in self.index]] <= 0).all()
        return (first_cond and sd_cond)

    def select_greatest_increase(self, theta, s, q, Q, S):
		#Sélectionne la composante dont l'ajout / la suppression / la réestimation
		#impliquerait la plus forte augmentation dans la vraisemblance marginale
	
        #Formules dans Appendix (A.2, A.3, A.4) : http://www.miketipping.com/papers/met-fastsbl.pdf
        
        to_reestimate = [i for i in range(self.M) if (theta[i]> 0) and (i in self.index)]
        to_add = [i for i in range(self.M) if theta[i] > 0 and not(i in self.index)]
        to_delete = [i for i in range(self.M) if (theta[i] <= 0) and (i in self.index)]
        
        delta_l_alpha = np.zeros(self.M)
        
        delta_l_alpha[to_add] = (Q[to_add]**2 - S[to_add])/S[to_add] + np.log(S[to_add]/Q[to_add]**2)
        alpha_updated_to_reestimate = s[to_reestimate]**2/(q[to_reestimate]**2-s[to_reestimate])
        alpha_inv_diff_to_reestimate = (1/alpha_updated_to_reestimate) - (1/self.alpha[to_reestimate])
        delta_l_alpha[to_reestimate] = Q[to_reestimate]**2/(S[to_reestimate] + (1/alpha_inv_diff_to_reestimate)) - np.log(1 + S[to_reestimate]*alpha_inv_diff_to_reestimate)
        delta_l_alpha[to_delete] = Q[to_delete]**2/(S[to_delete] - self.alpha[to_delete]) - np.log(1 - S[to_delete]/self.alpha[to_delete])
        
        return np.argmax(delta_l_alpha)
    
    def fit_init(self, X, y):
		#Initialise l'apprentissage du modèle RVM en initialisant les valeurs
        
        #initialisation du kernel
        if self.kernel==gaussian_kernel:
            self.kernel = partial(self.kernel, sigma=self.kernel_param)
        else:
            self.kernel = self.kernel
        
        self.phi = np.c_[np.ones(X.shape[0]), self.kernel(X, X)] #equation 19
        self.N = X.shape[0]
        self.M = (self.phi).shape[1]

        # Step 1 : initialise sigma selon les données d'entrainement
		#(voir steps dans article http://www.miketipping.com/papers/met-fastsbl.pdf)
        self.beta = 1 / (0.1 * np.var(y))

        # Step 2 : initialise le modèle avec une unique composante

        # Vecteur des alpha_i à infini
        self.alpha = np.ones(self.M) * np.inf

        # Sauf un initialisé equation 26 (le biais en position 0)
        index_init = 0
        norm_phi_square = np.linalg.norm(self.phi[:,index_init], ord=2)**2
        norm_phi_y_square = np.dot(self.phi[:,index_init].T, y)**2

        self.alpha[index_init] = norm_phi_square / ((norm_phi_y_square/norm_phi_square) - (1/self.beta))

        # index des alpha_i non infini
        self.index = [index_init]
        
        # Step 3 : calcule les paramètres associés une première fois

        A = self.compute_A()

        self.compute_sigma_and_mu(A, y)
        
        S, Q, s, q = self.compute_S_and_Q(A, y)
        
        return S, Q, s, q
        

    def fit(self, X, y):
		#Réalise l'apprentissage du RVM - en pratique l'apprentissage itératif des hyperparamètres alpha et beta
        
		#Initialisation des paramètres de l'apprentissage
        S, Q, s, q = self.fit_init(X, y)

        iter=0
        
        theta = np.inf*np.ones(self.M)

        while (iter < self.iter_max): # and not(self.check_convergence(theta, s, q)):

            # step 9 : réestime le paramètre de bruit beta
            self.recompute_beta(y)

            # step 4 : sélectionne une composante candidate
            if self.selection == "aleat":
                i = np.random.randint(self.M) #sélection aléatoire
            elif self.selection == "list":
                i = iter % self.M #sélection ordonnée
            else :
                i = self.select_greatest_increase(theta, s, q, Q, S) #sélection du maximiseur de la vraisemblance

            # step 5 : calcule theta pour décider ce qu'il faut faire de la composante sélectionnée
            theta_i = theta[i]

            # step 6 : réestime les valeurs associées à la composante s'il le faut
            if (theta_i > 0) and (i in self.index):
                self.reestimate(i, s, q)
            #step 7 : ajoute la composante au modèle s'il le faut
            elif (theta_i > 0) and not(i in self.index):
                self.add(i, s, q)
            #step 8 : enlève la composante du modèle s'il le faut
            elif (theta_i <= 0) and (i in self.index):
                if len(self.index) > 1:
                    self.delete(i)

            # step 10 : recalcule les paramètres de la loi a posteriori sigma et mu
            A = self.compute_A()
            self.compute_sigma_and_mu(A, y)
            S, Q, s, q = self.compute_S_and_Q(A, y)
            theta = q**2 - s

            if iter%100==0:
                print("iter:", iter)
                print("nb support vector:",len(self.index))

            iter+=1

        self.iter_total += iter

        self.alerte = False
        l = []
        for x in self.index:
            if x==0:
                self.alerte = True
            else:
                l.append(x-1)
        self.vecteur_support = X[l]

        return self


    def predict(self, X):
		#Fait une prédiction pour des données
        if self.alerte:
            phi_pred = np.c_[np.ones(X.shape[0]), self.kernel(X, self.vecteur_support)]
        else:
            phi_pred = self.kernel(X, self.vecteur_support)
        y_pred = phi_pred @ self.mu
        return y_pred
    
    
    def predict_proba(self, X):
		#Retourne la loi de probabilité prédite pour des données
		#Plus particulièrement ici, la moyenne et la variance de la loi, et les q95%
		
        if self.alerte:
            phi_star = np.c_[np.ones(X.shape[0]), self.kernel(X, self.vecteur_support)]
        else:
            phi_star = self.kernel(X, self.vecteur_support)

        mu_star = phi_star @ self.mu
        sigma_star = np.diag(1/self.beta + phi_star @ self.sigma @ phi_star.T)**0.5

        L_95 = mu_star - 2*sigma_star
        H_95 = mu_star + 2*sigma_star

        return mu_star, sigma_star, L_95, H_95

