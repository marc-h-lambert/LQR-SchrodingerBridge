import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
from scipy.stats import multivariate_normal
from matplotlib.pyplot import figure

###################################################################
# Gaussian-LQR Schrodinger Bridge where marginals are Gaussians
# and we suppose the reference measure is a LQR loss:
# ell(xk,xk+1)=(xk-xk_star)^T Qk (xk-xk_star)
#             +   (xk+1-xk)^T Rk (xk+1-xk)
###################################################################

plotDebug=False

class graphix:
    # plot a 2D ellipsoid
    def plot_ellipsoid2d(ax,origin,Cov,col='r',zorder=1,label='',linestyle='dashed',linewidth=1):
        L=LA.cholesky(Cov)
        theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
        x = np.cos(theta)
        y = np.sin(theta)
        x,y=origin.reshape(2,1) + L.dot([x, y])
        ax.plot(x, y,linestyle=linestyle,color=col,zorder=zorder,label=label,linewidth=linewidth)

    def plot_process(ax,lqr_sb,Nsamples=0,step=1,plotQ=True):
        graphix.plot_ellipsoid2d(ax, lqr_sb.mu_0, lqr_sb.Sigma_0, col='r', zorder=3, linestyle='-', linewidth=2)
        ax.scatter(lqr_sb.mu_0[0], lqr_sb.mu_0[1], c='r', marker='*')
        graphix.plot_ellipsoid2d(ax, lqr_sb.mu_K, lqr_sb.Sigma_K, col='r', zorder=3, linestyle='-', linewidth=2)
        ax.scatter(lqr_sb.mu_K[0], lqr_sb.mu_K[1], c='r', marker='*')

        mean_traj = np.array(lqr_sb.mean_traj)

        #ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'b-', label='Geodesic')
        for i in  range(0,Nsamples):
            traj=np.array(lqr_sb.sampleTraj())
            ax.plot(traj[:, 0], traj[:, 1], 'k',linestyle='-.',linewidth=2)
        # for k in range(1,K,100):
        for k in range(1, lqr_sb.K, step):
            graphix.plot_ellipsoid2d(ax, lqr_sb.mean_traj[k], lqr_sb.cov_traj[k], col='b', zorder=3, linestyle='-',linewidth=0.5)
            if k==1 and plotQ:
                ax.scatter(lqr_sb.x_star[k][0], lqr_sb.x_star[k][1], c='g', marker='*', label='Potential Q',linewidth=5)
                #graphix.plot_ellipsoid2d(ax, lqr_sb.x_star[k], lqr_sb.Q[k], col='g', zorder=3, linestyle='--', linewidth=1)
            if k>1 and plotQ:
                ax.scatter(lqr_sb.x_star[k][0], lqr_sb.x_star[k][1], c='g', marker='*',linewidth=5)
                #graphix.plot_ellipsoid2d(ax, lqr_sb.x_star[k], lqr_sb.Q[k], col='g', zorder=3, linestyle='--',linewidth=1)

    def plot_samples(ax, lqr_sb, Nsamples=1):
        for i in  range(0,Nsamples):
            traj=np.array(lqr_sb.sampleTraj())
            ax.plot(traj[:, 0], traj[:, 1], 'k',linestyle='-.',linewidth=0.5)

class LQR_SB:
    def __init__(self,mu_0, Sigma_0, mu_K, Sigma_K, x_star, Q, R,eps=1,nbPasses=1):
        self.K=len(x_star)
        self.mu_0=mu_0
        self.Sigma_0=Sigma_0
        self.mu_K = mu_K
        self.Sigma_K = Sigma_K
        self.x_star=x_star
        self.Q=Q
        self.R=R
        self.nbPasses=nbPasses
        self.eps=eps

        # default values of potentials
        self.alpha_0 = mu_0
        self.P_0 = LA.inv(Sigma_0)
        self.alpha_K = mu_K
        self.P_K = LA.inv(Sigma_K)

    def computePotentials(self):
        print("----- initialize FINAL potential as alpha_K= {} and P_K= {} -----".format(self.alpha_K, self.P_K))
        for numPass in range(0, self.nbPasses):
            # --- First Backward Pass: Compute suboptimal policy and cost-to-go ---
            if plotDebug:
                print("----- Backward pass numero {} ----".format(numPass + 1))
            alpha_list, P_list = LQR_SB.backward_potential(self.alpha_K, self.P_K, self.Q, self.x_star, self.R,self.eps)
            alpha_0 = alpha_list[0]
            P_0 = P_list[0]
            if plotDebug:
                print("----- propagate INITIAL potential is alpha_0= {} and P_0= {} -----".format(alpha_0, P_0))

            # Compute updated initial condition for forward pass
            self.P_0 = LA.inv(self.Sigma_0) - P_list[0]
            self.alpha_0 = LA.inv(self.P_0) @ (LA.inv(self.Sigma_0) @ self.mu_0 - P_list[0] @ alpha_list[0])
            if plotDebug:
                print("----- update INITIAL potential is alpha_0= {} and P_0= {} -----".format(self.alpha_0, self.P_0))

            # --- Forward Pass: Propagate potential and update alpha, P ---
            if plotDebug:
                print("----- Forward pass numero {} ----".format(numPass + 1))
            alpha_list, P_list = LQR_SB.forward_potential(self.alpha_0, self.P_0, self.Q, self.x_star, self.R,self.eps)
            alpha_K = alpha_list[-1]
            P_K = P_list[-1]
            if plotDebug:
                print("----- propagate FINAL potential is alpha_K= {} and P_K= {} -----".format(alpha_K, P_K))

            # Compute updated terminal condition for second backward pass
            self.P_K = LA.inv(self.Sigma_K) - P_list[-1]
            self.alpha_K = LA.inv(self.P_K) @ (LA.inv(self.Sigma_K) @ self.mu_K - P_list[-1] @ alpha_list[-1])
            if plotDebug:
                print("----- update FINAL potential is alpha_K= {} and P_K= {} -----".format(self.alpha_K, self.P_K))

    def computePotentials_plot(self,axsBackward,axsForward):
        print("----- initialize FINAL potential as alpha_K= {} and P_K= {} -----".format(self.alpha_K, self.P_K))
        iSigma_0 = LA.inv(self.Sigma_0)
        iSigma_K = LA.inv(self.Sigma_K)
        for numPass in range(0, self.nbPasses):
            # --- First Backward Pass: Compute suboptimal policy and cost-to-go ---
            if plotDebug:
                print("----- Backward pass numero {} ----".format(numPass + 1))
            alpha_list, P_list = LQR_SB.backward_potential(self.alpha_K, self.P_K, self.Q, self.x_star, self.R,self.eps)
            alpha_list = np.array(alpha_list)
            P_list = np.array(P_list)
            #P_list=P_list + self.P_0
            axsBackward[0,0].plot(range(0, K + 1), alpha_list[:, 0],label="pass {}".format(numPass))
            axsBackward[0, 0].scatter(0, self.mu_0[0],color='red',marker='*')
            axsBackward[0, 0].scatter(K, self.mu_K[0], color='red', marker='*')
            axsBackward[0,1].plot(range(0, K + 1), alpha_list[:, 1], label="pass {}".format(numPass))
            axsBackward[0, 1].scatter(0, self.mu_0[1], color='red', marker='*')
            axsBackward[0, 1].scatter(K, self.mu_K[1], color='red', marker='*')
            axsBackward[1,0].plot(range(0, K + 1), P_list[:, 0,0], label="pass {}".format(numPass))
            axsBackward[1, 0].scatter(0, iSigma_0[0,0], color='red', marker='*')
            axsBackward[1, 0].scatter(K, iSigma_K[0,0], color='red', marker='*')
            axsBackward[1,1].plot(range(0, K + 1), P_list[:, 1,1], label="pass {}".format(numPass))
            axsBackward[1, 1].scatter(0, iSigma_0[1,1], color='red', marker='*')
            axsBackward[1, 1].scatter(K, iSigma_K[1,1], color='red', marker='*')

            alpha_0_minus = alpha_list[0]
            P_0_minus = P_list[0]
            if plotDebug:
                print("----- propagate INITIAL potential is alpha_0= {} and P_0= {} -----".format(alpha_0_minus, P_0_minus))

            # Compute updated initial condition for forward pass
            self.P_0 = LA.inv(self.Sigma_0) - P_0_minus
            self.alpha_0 = LA.inv(self.P_0) @ (LA.inv(self.Sigma_0) @ self.mu_0 - P_0_minus @ alpha_0_minus)
            if plotDebug:
                print("----- update INITIAL potential is alpha_0= {} and P_0= {} -----".format(self.alpha_0, self.P_0))

            # --- Forward Pass: Propagate potential and update alpha, P ---
            if plotDebug:
                print("----- Forward pass numero {} ----".format(numPass + 1))
            alpha_list, P_list = LQR_SB.forward_potential(self.alpha_0, self.P_0, self.Q, self.x_star, self.R,self.eps)
            alpha_list = np.array(alpha_list)
            P_list = np.array(P_list)
            axsForward[0, 0].plot(range(0, K + 1), alpha_list[:, 0], label="pass {}".format(numPass))
            axsForward[0, 0].scatter(0, self.mu_0[0], color='red', marker='*')
            axsForward[0, 0].scatter(K, self.mu_K[0], color='red', marker='*')
            axsForward[0, 1].plot(range(0, K + 1), alpha_list[:, 1], label="pass {}".format(numPass))
            axsForward[0, 1].scatter(0, self.mu_0[1], color='red', marker='*')
            axsForward[0, 1].scatter(K, self.mu_K[1], color='red', marker='*')
            axsForward[1, 0].plot(range(0, K + 1), P_list[:, 0, 0], label="pass {}".format(numPass))
            axsForward[1, 0].scatter(0, self.Sigma_0[0, 0], color='red', marker='*')
            axsForward[1, 0].scatter(K, self.Sigma_K[0, 0], color='red', marker='*')
            axsForward[1, 1].plot(range(0, K + 1), P_list[:, 1, 1], label="pass {}".format(numPass))
            axsForward[1, 1].scatter(0, self.Sigma_0[1, 1], color='red', marker='*')
            axsForward[1, 1].scatter(K, self.Sigma_K[1, 1], color='red', marker='*')

            alpha_K_plus = alpha_list[-1]
            P_K_plus = P_list[-1]
            if plotDebug:
                print("----- propagate FINAL potential is alpha_K= {} and P_K= {} -----".format(alpha_K_plus, P_K_plus))

            # Compute updated terminal condition for second backward pass
            self.P_K = LA.inv(self.Sigma_K) - P_K_plus
            self.alpha_K = LA.inv(self.P_K) @ (LA.inv(self.Sigma_K) @ self.mu_K - P_K_plus @ alpha_K_plus)
            if plotDebug:
                print("----- update FINAL potential is alpha_K= {} and P_K= {} -----".format(self.alpha_K, self.P_K))



    def computeProcess(self):
        alpha_list, P_list, self.policy_bias, self.policy_gain, self.policy_cov = LQR_SB.backward_policy(self.alpha_K, self.P_K, self.Q, self.x_star, self.R,self.eps)
        self.mean_traj, self.cov_traj = LQR_SB.forward_marginal(self.mu_0, self.Sigma_0, self.policy_bias, self.policy_gain, self.policy_cov)

    def sampleTraj(self):
        return LQR_SB.forward_traj(self.mu_0, self.Sigma_0, self.policy_bias,self.policy_gain, self.policy_cov)

    ###################################################################
    # backward_potential
    # Starting from a final quadratic cost VK
    # we build the final Gaussian potential exp(-VK) propto N(alphaK,PK)
    # and propagate it from t=K to t=0 through LQR ref measure
    # to compute intermediate potentials exp(-Vk) propto N(alphak,Pk)
    ###################################################################
    @staticmethod
    def backward_potential(alpha_K, P_K, Q, x_star, R,eps):
        K=len(x_star)
        alpha_k = alpha_K
        P_k = P_K
        alpha_list = []
        P_list = []
        alpha_list.insert(0,alpha_k)
        P_list.insert(0,P_k)

        for k in range(K - 1, -1,-1):
            if plotDebug:
                print("Backward pass - iteration number {}".format(k))
            P_k = Q[k]/eps + P_k - P_k @ LA.inv(R[k]/eps + P_k) @ P_k
            alpha_k = alpha_k + LA.inv(P_k) @ (Q[k]/eps) @ (x_star[k] - alpha_k)

            alpha_list.insert(0,alpha_k)
            P_list.insert(0,P_k)

            if plotDebug:
                print("alpha_k=", alpha_k)
                print("P_k=", P_k)

        return alpha_list, P_list


    ###################################################################
    # forward_potential
    # Starting from a newly update cost V0
    # we build the Gaussian potential exp(-V0) propto N(alpha0,P0)
    # and propagate it from t=0 to t=K through LQR ref measure
    # to compute intermediate potentials exp(-Vk) propto N(alphak,Pk)
    ###################################################################
    @staticmethod
    def forward_potential(alpha_0, P_0,Q, x_star, R, eps):
        K=len(x_star)
        alpha_k = alpha_0
        P_k = P_0
        alpha_list = []
        P_list = []
        alpha_list.append(alpha_k)
        P_list.append(P_k)

        for k in range(0,K,1):
            if plotDebug:
                print("Forward pass - iteration number {}".format(k))
            alpha_k = LA.inv(Q[k]/eps + P_k) @ (Q[k] @ x_star[k]/eps + P_k @ alpha_k)
            P_k = LA.inv(eps*LA.inv(R[k]) + LA.inv(Q[k]/eps + P_k))

            alpha_list.append(alpha_k)
            P_list.append(P_k)

            if plotDebug:
                print("alpha_k=", alpha_k)
                print("P_k=", P_k)

        return alpha_list, P_list

    ###################################################################
    # forward_potential
    # Starting from a newly update cost V0
    # we build the Gaussian potential exp(-V0) propto N(alpha0,P0)
    # and propagate it from t=0 to t=K through LQR ref measure
    # to compute intermediate potentials exp(-Vk) propto N(alphak,Pk)
    ###################################################################
    @staticmethod
    def forward_potential_Dirac(delta_0, Q, x_star, R):
        K=len(x_star)
        alpha_k = delta_0
        P_k = R[0]
        alpha_list = []
        P_list = []
        alpha_list.append(alpha_k)
        P_list.append(P_k)
        alpha_list.append(alpha_k)
        P_list.append(P_k)

        for k in range(1,K,1):
            if plotDebug:
                print("Forward pass - iteration number {}".format(k))
            alpha_k = LA.inv(Q[k]/eps + P_k) @ (Q[k] @ x_star[k]/eps + P_k @ alpha_k)
            P_k = LA.inv(eps*LA.inv(R[k]) + LA.inv(Q[k]/eps + P_k))

            alpha_list.append(alpha_k)
            P_list.append(P_k)

            if plotDebug:
                print("alpha_k=", alpha_k)
                print("P_k=", P_k)

        return alpha_list, P_list

    ###################################################################
    # backward_policy
    # Identical to backward_potential but we compute also the
    # optimal policy p(uk|xk)=N(beta_k + K_k.xk ,iS_k) on the path
    # the potential V0 and VK are supposed to have converged
    # before calling this
    ###################################################################
    @staticmethod
    def backward_policy(alpha_K, P_K,Q, x_star, R, eps):
        K=len(x_star)
        policy_cov = []
        policy_gain = []
        policy_bias = []
        alpha_k = alpha_K
        P_k = P_K
        alpha_list = []
        P_list = []
        alpha_list.insert(0, alpha_k)
        P_list.insert(0, P_k)

        for k in range(K - 1, -1,-1):
            # Update policy
            S_k = R[k]/eps + P_k
            iS_k=LA.inv(S_k)
            Beta_k=iS_k @ P_k @ alpha_k
            Gain_k=-iS_k @ P_k
            Cov_k = iS_k
            policy_bias.insert(0,Beta_k)
            policy_gain.insert(0,Gain_k)
            policy_cov.insert(0,Cov_k)

            # Update cost-to-go
            P_k = Q[k]/eps + P_k - P_k @ LA.inv(R[k]/eps + P_k) @ P_k
            alpha_k = alpha_k + LA.inv(P_k) @ (Q[k]/eps) @ (x_star[k] - alpha_k)
            if plotDebug:
                print("alpha_k=", alpha_k)
                print("P_k=", P_k)
            alpha_list.insert(0, alpha_k)
            P_list.insert(0, P_k)

        return alpha_list, P_list, policy_bias,policy_gain,policy_cov



    ###################################################################
    # forward_traj
    # We compute here the closed loop optimal trajectory
    # as Gaussian marginals N(mu_k,Sigma_k) from k=0 to k=K
    # starting from the prior marginal N(mu_0,Sigma_0)
    # and hopefully ending at the final marginal N(mu_K,Sigma_K)
    # (we must end to it if the potentials V0 and VK have converged)
    ###################################################################
    @staticmethod
    def forward_marginal(mu_0, Sigma_0,policy_bias,policy_gain,policy_cov):
        K = len(policy_bias)
        dt=1
        d=mu_0.shape[0]
        mu_k = mu_0
        Sigma_k = Sigma_0

        mean_traj = []
        cov_traj = []

        mean_traj.append(mu_k)
        cov_traj.append(Sigma_k)

        if plotDebug:
            print("mu_k=", mu_k)
            print("Sigma_k=", Sigma_k)
        for k in range(0,K,1):
            A=np.identity(d)+dt *policy_gain[k]
            iSk=policy_cov[k]
            b=dt*policy_bias[k]
            mu_k = b + A @ mu_k
            Sigma_k = iSk + A @ Sigma_k @ A.T

            if plotDebug:
                print("iSk=", iSk)
                print("A=", A)
                print("mu_k=",mu_k)
                print("Sigma_k=", Sigma_k)

            mean_traj.append(mu_k)
            cov_traj.append(Sigma_k)

        return mean_traj, cov_traj

    @staticmethod
    def forward_traj(mu_0, Sigma_0, policy_bias, policy_gain, policy_cov):
        K = len(policy_bias)
        d = mu_0.shape[0]

        traj=[]
        x_0 = np.random.multivariate_normal(mu_0,Sigma_0)
        x_k=x_0
        traj.append(x_k)

        for k in range(0, K, 1):
            x_k = np.random.multivariate_normal(policy_bias[k] + x_k + policy_gain[k] @ x_k, policy_cov[k])
            traj.append(x_k)

        return traj

    @staticmethod
    def forward_marginal_Dirac(delta0,R0,policy_bias,policy_gain,policy_cov):
        K = len(policy_bias)
        dt=1
        d=delta0.shape[0]
        mu_k = delta0
        Sigma_k = R0

        mean_traj = []
        cov_traj = []

        mean_traj.append(mu_k)
        cov_traj.append(Sigma_k)
        mean_traj.append(mu_k)
        cov_traj.append(Sigma_k)

        for k in range(1,K,1):
            A=np.identity(d)+dt *policy_gain[k]
            iSk=policy_cov[k]
            b=dt*policy_bias[k]
            mu_k = b + A @ mu_k
            Sigma_k = iSk + A @ Sigma_k @ A.T

            if plotDebug:
                print("iSk=", iSk)
                print("A=", A)
                print("mu_k=",mu_k)
                print("Sigma_k=", Sigma_k)

            mean_traj.append(mu_k)
            cov_traj.append(Sigma_k)

        return mean_traj, cov_traj

if __name__ == "__main__":

    #TEST = ["Test0"]
    TEST=["Test1-1","Test1-2","Test1-3","Test1-4"]
    #TEST = ["Test2-1", "Test2-2", "Test2-3"]
    #TEST = ["Test2-4"]
    #TEST = ["Test3-1","Test3-2"]

    if "Test0" in TEST:
        print("----- Initialization ----")
        K = 15  # Number of time steps
        nbPasses = 5
        dt = 1
        d = 2  # Dimension of the state space
        eps = 1e-3  # for eps smaller the convergence is slower, increase the number of passes
        r = 1  # r=1: Bures / r<1 Entropic Bures
        q = 0.02

        # Define Gaussian marginals
        mu_0 = np.array([0.0, 0.0])
        Sigma_0 = 0.01*np.identity(d)

        mu_K = np.array([2.0, 2.0])
        Sigma_K = 0.03*np.identity(d)

        # Define Q and R matrices
        Q = [q* dt * np.identity(d) for i in range(K)]
        R = [r * np.identity(d) for _ in range(K)]

        # Define desired trajectory
        x_star = [np.array([2.0, 0.0]) for i in range(K)]

        lqr_sb=LQR_SB(mu_0, Sigma_0, mu_K, Sigma_K, x_star, Q, R,eps,nbPasses=nbPasses)

        fig, axs= plt.subplots(2,2,figsize=(8, 6))
        fig2, axs2 = plt.subplots(2, 2, figsize=(8, 6))
        lqr_sb.computePotentials_plot(axs,axs2)
        #plt.title("convergence of alpha0")
        axs[0,0].legend()
        #axs[0,1].legend()
        #axs[1,0].legend()
        #axs[1,1].legend()
        axs[0, 0].set_title("alpha[{}]".format(0))
        axs[0,1].set_title("alpha[{}]".format(1))
        axs[1,0].set_title("P[{}{}]".format(0,0))
        axs[1,1].set_title("P[{}{}]".format(1,1))
        plt.savefig("PotentialEvolutionBackward.png")



        axs2[0, 0].legend()
        axs2[0, 0].set_title("alpha[{}]".format(0))
        axs2[0, 1].set_title("alpha[{}]".format(1))
        axs2[1, 0].set_title("P[{}{}]".format(0, 0))
        axs2[1, 1].set_title("P[{}{}]".format(1, 1))
        plt.savefig("PotentialEvolutionForward.png")
        plt.show()

        #lqr_sb.computeProcess()
        print("----- we start from the Gaussian marginal mu_0= {} and Sigma_0= {} -----".format(mu_0, Sigma_0))
        print("----- we end ate the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(lqr_sb.mean_traj[-1], lqr_sb.cov_traj[-1]))
        print("----- we should end at the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(mu_K, Sigma_K))
        # if the Gaussian marginals does not fit that probably mean the potentials V0 and VK have not well converged

        # Visualization
        fig, ax = plt.subplots()
        #graphix.plot_process(ax, lqr_sb,plotQ=False)
        ax.set_xlim(-0.5, 3)
        ax.set_ylim(-1, 3)
        ax.legend( fontsize=12)
        plt.savefig("Test1_1.png")
        plt.show()

    if "Test1-1" in TEST:
        print("----- Initialization ----")
        K = 15  # Number of time steps
        nbPasses=5
        dt=1
        d = 2  # Dimension of the state space
        eps=1e-3 # for eps smaller the convergence is slower, increase the number of passes
        r=1 # r=1: Bures / r<1 Entropic Bures
        q=0

        # Define Gaussian marginals
        mu_0 = np.array([0.0, 0.0])
        Sigma_0 = 0.01*np.identity(d)

        mu_K = np.array([2.0, 2.0])
        Sigma_K = 0.03*np.array([[1.0, 0.5],[0.5, 2.0]])

        # Define Q and R matrices
        Q = [q* dt * np.identity(d) for i in range(K)]
        R = [r * np.identity(d) for _ in range(K)]

        # Define desired trajectory
        x_star = [np.array([2.0, 0.0]) for i in range(K)]

        lqr_sb=LQR_SB(mu_0, Sigma_0, mu_K, Sigma_K, x_star, Q, R,eps,nbPasses=nbPasses)
        lqr_sb.computePotentials()
        lqr_sb.computeProcess()
        print("----- we start from the Gaussian marginal mu_0= {} and Sigma_0= {} -----".format(mu_0, Sigma_0))
        print("----- we end ate the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(lqr_sb.mean_traj[-1], lqr_sb.cov_traj[-1]))
        print("----- we should end at the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(mu_K, Sigma_K))
        # if the Gaussian marginals does not fit that probably mean the potentials V0 and VK have not well converged

        # Visualization
        fig, ax = plt.subplots()
        graphix.plot_process(ax, lqr_sb,plotQ=False)
        ax.set_xlim(-0.5, 3)
        ax.set_ylim(-1, 3)
        ax.legend( fontsize=12)
        plt.savefig("Test1_1.png")
        plt.show()

    if "Test1-2" in TEST:
        print("----- Initialization ----")
        K = 15  # Number of time steps
        nbPasses = 3
        dt = 1
        d = 2  # Dimension of the state space
        eps = 1e-3  # for eps smaller the convergence is slower, increase the number of passes
        r = 1  # r=1: Bures / r<1 Entropic Bures
        q = 0.02

        # Define Gaussian marginals
        mu_0 = np.array([0.0, 0.0])
        Sigma_0 = 0.01*np.identity(d)

        mu_K = np.array([2.0, 2.0])
        Sigma_K = 0.03*np.array([[1.0, 0.5],[0.5, 2.0]])

        # Define Q and R matrices
        Q = [q* dt * np.identity(d) for i in range(K)]
        R = [r * np.identity(d) for _ in range(K)]

        # Define desired trajectory
        x_star = [np.array([2.0, 0.0]) for i in range(K)]

        lqr_sb=LQR_SB(mu_0, Sigma_0, mu_K, Sigma_K, x_star, Q, R,eps,nbPasses=nbPasses)
        lqr_sb.computePotentials()
        lqr_sb.computeProcess()
        print("----- we start from the Gaussian marginal mu_0= {} and Sigma_0= {} -----".format(mu_0, Sigma_0))
        print("----- we end ate the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(lqr_sb.mean_traj[-1], lqr_sb.cov_traj[-1]))
        print("----- we should end at the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(mu_K, Sigma_K))
        # if the Gaussian marginals does not fit that probably mean the potentials V0 and VK have not well converged

        # Visualization
        fig, ax = plt.subplots()
        graphix.plot_process(ax, lqr_sb, plotQ=True)
        ax.set_xlim(-0.5, 3)
        ax.set_ylim(-1, 3)
        ax.legend( fontsize=12)
        plt.savefig("Test1_2.png")
        plt.show()

    if "Test1-3" in TEST:
        print("----- Initialization ----")
        K = 15  # Number of time steps
        nbPasses = 30
        dt = 1
        d = 2  # Dimension of the state space
        eps = 1e-3  # for eps smaller the convergence is slower, increase the number of passes
        r = 1  # r=1: Bures / r<1 Entropic Bures
        q = 0.1

        # Define Gaussian marginals
        mu_0 = np.array([0.0, 0.0])
        Sigma_0 = 0.01*np.identity(d)

        mu_K = np.array([2.0, 2.0])
        Sigma_K = 0.03*np.array([[1.0, 0.5],[0.5, 2.0]])

        # Define Q and R matrices
        Q = [q* dt * np.identity(d) for i in range(K)]
        R = [r * dt * np.identity(d) for _ in range(K)]

        # Define desired trajectory
        x_star = [np.array([2.0, 0.0]) for i in range(K)]

        lqr_sb=LQR_SB(mu_0, Sigma_0, mu_K, Sigma_K, x_star, Q, R,eps)
        lqr_sb.computePotentials()
        lqr_sb.computeProcess()
        print("----- we start from the Gaussian marginal mu_0= {} and Sigma_0= {} -----".format(mu_0, Sigma_0))
        print("----- we end ate the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(lqr_sb.mean_traj[-1], lqr_sb.cov_traj[-1]))
        print("----- we should end at the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(mu_K, Sigma_K))
        # if the Gaussian marginals does not fit that probably mean the potentials V0 and VK have not well converged

        # Visualization
        fig, ax = plt.subplots()
        graphix.plot_process(ax, lqr_sb)
        ax.set_xlim(-0.5, 3)
        ax.set_ylim(-1, 3)
        ax.legend( fontsize=12)
        plt.savefig("Test1_3.png")
        plt.show()

    if "Test1-4" in TEST:
        print("----- Initialization ----")
        K = 50  # Number of time steps
        nbPasses = 30
        dt = 1
        d = 2  # Dimension of the state space
        eps = 1e-3  # for eps smaller the convergence is slower, increase the number of passes
        r = 10  # r=1: Bures / r<1 Entropic Bures
        q = 0.5

        # Define Gaussian marginals
        mu_0 = np.array([0.0, 0.0])
        Sigma_0 = 0.01*np.identity(d)

        mu_K = np.array([2.0, 2.0])
        Sigma_K = 0.03*np.array([[1.0, 0.5],[0.5, 2.0]])

        # Define Q and R matrices
        Q = [q* dt * np.identity(d) for i in range(K)]
        R = [r * dt * np.identity(d) for _ in range(K)]

        # Define desired trajectory
        x_star = [np.array([2.0, 0.0]) for i in range(K)]

        lqr_sb=LQR_SB(mu_0, Sigma_0, mu_K, Sigma_K, x_star, Q, R,eps)
        lqr_sb.computePotentials()
        lqr_sb.computeProcess()
        print("----- we start from the Gaussian marginal mu_0= {} and Sigma_0= {} -----".format(mu_0, Sigma_0))
        print("----- we end ate the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(lqr_sb.mean_traj[-1], lqr_sb.cov_traj[-1]))
        print("----- we should end at the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(mu_K, Sigma_K))
        # if the Gaussian marginals does not fit that probably mean the potentials V0 and VK have not well converged

        # Visualization
        fig, ax = plt.subplots()
        graphix.plot_process(ax, lqr_sb)
        ax.set_xlim(-0.5, 3)
        ax.set_ylim(-1, 3)
        ax.legend( fontsize=12)
        plt.savefig("Test1_4.png")
        plt.show()

    if "Test2-1" in TEST:
        print("----- Initialization ----")
        K = 100  # Number of time steps
        nbPasses = 1
        dt = 1
        d = 2  # Dimension of the state space
        eps = 1e-3  # for eps smaller the convergence is slower, increase the number of passes
        r = 10  # r=1: Bures / r<1 Entropic Bures
        q = 0.1

        # Define Gaussian marginals
        mu_0 = np.array([0.0, 0.0])
        Sigma_0 = 0.1 * np.array([[3.0, 0.7], [0.7, 0.2]])

        mu_K = np.array([10.0, 0.0])
        Sigma_K = 0.3 * np.array([[1.0, 0.5], [0.5, 2.0]])

        # Define Q and R matrices
        Q = [q* dt * np.identity(d) for i in range(K)]
        R = [r * dt * np.identity(d) for _ in range(K)]

        # Define desired trajectory
        x_star = [np.array([3.0, -4.0]) if i in range(0,int(K/2)) else np.array([6.0, 4.0]) for i in range(K)]

        lqr_sb=LQR_SB(mu_0, Sigma_0, mu_K, Sigma_K, x_star, Q, R,eps)
        lqr_sb.computePotentials()
        lqr_sb.computeProcess()
        print("----- we start from the Gaussian marginal mu_0= {} and Sigma_0= {} -----".format(mu_0, Sigma_0))
        print("----- we end ate the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(lqr_sb.mean_traj[-1], lqr_sb.cov_traj[-1]))
        print("----- we should end at the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(mu_K, Sigma_K))
        # if the Gaussian marginals does not fit that probably mean the potentials V0 and VK have not well converged

        # Visualization
        fig, ax = plt.subplots()
        graphix.plot_process(ax, lqr_sb)
        ax.set_xlim(-1, 11)
        ax.set_ylim(-5, 5)
        ax.legend( fontsize=12)
        plt.title("Zig-zag", fontweight="bold", fontsize=16)
        plt.savefig("ZigZag.png")
        plt.show()

    if "Test2-2" in TEST:
        print("----- Initialization ----")
        K = 200  # Number of time steps
        nbPasses = 1
        dt = 1
        d = 2  # Dimension of the state space
        eps = 1e-3  # for eps smaller the convergence is slower, increase the number of passes
        r = 10  # r=1: Bures / r<1 Entropic Bures
        q = 0.2

        # Define Gaussian marginals
        mu_0 = np.array([0.0, 0.0])
        Sigma_0 = 0.5 * np.identity(2)

        mu_K = np.array([10.0, 0.0])
        Sigma_K = 0.5 * np.identity(2)

        # Define Q and R matrices
        Q = [q *  np.array([[1.0, -1], [-1, 2]]) for i in range(K)]
        R = [r *  np.identity(d) for _ in range(K)]

        # Define desired trajectory
        x_star = [np.array([5.0, 0.0])  for i in range(K)]

        lqr_sb=LQR_SB(mu_0, Sigma_0, mu_K, Sigma_K, x_star, Q, R,eps)
        lqr_sb.computePotentials()
        lqr_sb.computeProcess()
        print("----- we start from the Gaussian marginal mu_0= {} and Sigma_0= {} -----".format(mu_0, Sigma_0))
        print("----- we end ate the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(lqr_sb.mean_traj[-1], lqr_sb.cov_traj[-1]))
        print("----- we should end at the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(mu_K, Sigma_K))
        # if the Gaussian marginals does not fit that probably mean the potentials V0 and VK have not well converged

        # Visualization
        fig, ax = plt.subplots()
        graphix.plot_process(ax, lqr_sb)
        ax.set_xlim(-1, 11)
        ax.set_ylim(-5, 5)
        ax.legend( fontsize=12)
        plt.title("Twister", fontweight="bold", fontsize=16)
        plt.savefig("Twister.png")
        plt.show()

    if "Test2-3" in TEST:
        print("----- Initialization ----")
        K = 100  # Number of time steps
        nbPasses = 1
        dt = 1
        d = 2  # Dimension of the state space
        eps = 1e-3  # for eps smaller the convergence is slower, increase the number of passes
        r = 10  # r=1: Bures / r<1 Entropic Bures
        q = 0.2

        # Define Gaussian marginals
        mu_0 = np.array([0.0, 0.0])
        Sigma_0 = 0.5 * np.identity(2)

        mu_K = np.array([10.0, 0.0])
        Sigma_K = 0.5 * np.identity(2)

        # Define Q and R matrices
        Q = [q * np.identity(d) if i in range(0,int(K/2)) else 1e-6 * np.identity(d) for i in range(K)]
        R = [r *  np.identity(d) for _ in range(K)]

        # Define desired trajectory
        x_star = [np.array([4.0, 2]) if i in range(0, int(K / 6)) else np.array([2.0, 5]) if i in range(int(K / 6),int(K / 3)) else np.array([.0, 2]) for i in range(K)]

        lqr_sb=LQR_SB(mu_0, Sigma_0, mu_K, Sigma_K, x_star, Q, R,eps)
        lqr_sb.computePotentials()
        lqr_sb.computeProcess()
        print("----- we start from the Gaussian marginal mu_0= {} and Sigma_0= {} -----".format(mu_0, Sigma_0))
        print("----- we end ate the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(lqr_sb.mean_traj[-1], lqr_sb.cov_traj[-1]))
        print("----- we should end at the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(mu_K, Sigma_K))
        # if the Gaussian marginals does not fit that probably mean the potentials V0 and VK have not well converged

        # Visualization
        fig, ax = plt.subplots()
        graphix.plot_process(ax, lqr_sb)
        ax.set_xlim(-1, 11)
        ax.set_ylim(-2, 9)
        ax.legend( fontsize=12)
        plt.title("Scoubidou", fontweight="bold", fontsize=16)
        plt.savefig("Scoubidou.png")
        plt.show()

    if "Test2-4" in TEST:
        print("----- Initialization ----")
        K = 15  # Number of time steps
        nbPasses = 1
        dt = 1
        d = 2  # Dimension of the state space
        #eps = 1e-3  # for eps smaller the convergence is slower, increase the number of passes
        r = 1  # r=1: Bures / r<1 Entropic Bures
        q = 10
        #######  divide matrix Q by 50 in plot elliposid above for better visu ######

        # Define Gaussian marginals
        mu_0 = np.array([0.0, 0.0])
        Sigma_0 = 0.1 * np.identity(2)

        mu_K = np.array([10.0, 0.0])
        Sigma_K = 0.1 * np.identity(2)

        # Define Q and R matrices
        Q = [q *  np.identity(d) for i in range(K)]
        R = [r *  np.identity(d) for _ in range(K)]

        # Define desired trajectory
        x_star = [np.array([5.0, 0.0])  for i in range(K)]
        x_star=np.array(x_star)
        x_star[0]=mu_0
        dt=10/K
        for i in range(1,K):
            x_star[i,0]=x_star[i-1,0]+dt
            x_star[i, 1]=math.sin(x_star[i, 0])

        lqr_sb=LQR_SB(mu_0, Sigma_0, mu_K, Sigma_K, x_star, Q, R,eps)
        lqr_sb.computePotentials()
        lqr_sb.computeProcess()
        print("----- we start from the Gaussian marginal mu_0= {} and Sigma_0= {} -----".format(mu_0, Sigma_0))
        print("----- we end ate the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(lqr_sb.mean_traj[-1], lqr_sb.cov_traj[-1]))
        print("----- we should end at the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(mu_K, Sigma_K))
        # if the Gaussian marginals does not fit that probably mean the potentials V0 and VK have not well converged

        # Visualization
        fig, ax = plt.subplots()
        graphix.plot_process(ax, lqr_sb)
        ax.set_xlim(-1, 11)
        ax.set_ylim(-5, 5)
        ax.legend( fontsize=12)
        plt.title("Wave", fontweight="bold", fontsize=16)
        plt.savefig("Wave.png")
        plt.show()

    if "Test3-1" in TEST:
        print("----- Initialization ----")
        K = 10  # Number of time steps
        nbPasses=1
        dt=1
        d = 2  # Dimension of the state space
        eps=1 # for eps smaller the convergence is slower, increase the number of passes
        r=100 # r=1: Bures / r<1 Entropic Bures
        q=0
        seed=1
        np.random.seed(seed)

        # Define Gaussian marginals
        mu_0 = np.array([0.0, 0.0])
        Sigma_0 = 1e-4*np.identity(d)

        mu_K = np.array([2.0, 2.0])
        Sigma_K = 1e-4*np.identity(d)

        # Define Q and R matrices
        Q = [q* dt * np.identity(d) for i in range(K)]
        R = [r * np.identity(d) for _ in range(K)]

        # Define desired trajectory
        x_star = [np.array([1.0, 1.0]) for i in range(K)]

        lqr_sb=LQR_SB(mu_0, Sigma_0, mu_K, Sigma_K, x_star, Q, R,eps)
        lqr_sb.computePotentials()
        lqr_sb.computeProcess()
        print("----- we start from the Gaussian marginal mu_0= {} and Sigma_0= {} -----".format(mu_0, Sigma_0))
        print("----- we end ate the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(lqr_sb.mean_traj[-1], lqr_sb.cov_traj[-1]))
        print("----- we should end at the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(mu_K, Sigma_K))
        # if the Gaussian marginals does not fit that probably mean the potentials V0 and VK have not well converged

        # Visualization
        fig, ax = plt.subplots()
        graphix.plot_process(ax, lqr_sb,plotQ=False,Nsamples=5)
        #raphix.plot_samples(ax, lqr_sb,Nsamples=1)
        ax.set_xlim(-0.5, 3)
        ax.set_ylim(-1, 3)
        ax.legend( fontsize=12)
        plt.savefig("BrownianBridge.png")
        plt.show()

    if "Test3-2" in TEST:
        print("----- Initialization ----")
        K = 20  # Number of time steps
        nbPasses=1
        dt=1
        d = 2  # Dimension of the state space
        eps=1e-3 # for eps smaller the convergence is slower, increase the number of passes
        r=10 # r=1: Bures / r<1 Entropic Bures
        q=0.3
        seed = 1
        np.random.seed(seed)

        # Define Gaussian marginals
        mu_0 = np.array([0.0, 0.0])
        Sigma_0 = 0.03*np.identity(d)

        mu_K = np.array([2.0, 2.0])
        Sigma_K = 0.03*np.identity(d)

        # Define Q and R matrices
        Q = [q* dt * np.identity(d) for i in range(K)]
        R = [r * np.identity(d) for _ in range(K)]

        # Define desired trajectory
        x_star = [np.array([1.0, 1.0]) for i in range(K)]

        lqr_sb=LQR_SB(mu_0, Sigma_0, mu_K, Sigma_K, x_star, Q, R,eps)
        lqr_sb.computePotentials()
        lqr_sb.computeProcess()
        print("----- we start from the Gaussian marginal mu_0= {} and Sigma_0= {} -----".format(mu_0, Sigma_0))
        print("----- we end ate the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(lqr_sb.mean_traj[-1], lqr_sb.cov_traj[-1]))
        print("----- we should end at the Gaussian marginal mu_K= {} and Sigma_K= {}  -----".format(mu_K, Sigma_K))
        # if the Gaussian marginals does not fit that probably mean the potentials V0 and VK have not well converged

        # Visualization
        fig, ax = plt.subplots()
        graphix.plot_process(ax, lqr_sb,plotQ=True,Nsamples=5)
        #raphix.plot_samples(ax, lqr_sb,Nsamples=1)
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        ax.legend( fontsize=12)
        plt.savefig("BrownianBridgeQ.png")
        plt.show()
