import torch
import gpytorch
import torch
import gpytorch
import numpy as np
import tqdm
class GaussianProcessRegressor(gpytorch.models.ExactGP):
    def __init__(self, X, y,likelihood):
        super().__init__(X,y,likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # self.X = torch.from_numpy(X).float()
        # self.y = torch.from_numpy(y).float()

    def forward(self, x):
        x = x.float()
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
import itertools
class GaussianProcessActiveLearner:
    def __init__(self, X_np, y_np,device='cpu'):
        self.X = torch.from_numpy(X_np).float().to(device)
        self.y = torch.from_numpy(y_np).float().to(device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.to(device)
        # self.model = self.create_model()
        self.model = GaussianProcessRegressor(self.X,self.y,self.likelihood)
        self.model.to(device)
    
    # def create_model(self):
    #     likelihood = self.likelihood
    #     model = gpytorch.models.ExactGP(self.X, self.y, likelihood)
    #     model.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    #     model.mean_module = gpytorch.means.ConstantMean()
    #     return model
    
    def fit(self, iter):
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(),self.likelihood.parameters()), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in tqdm.tqdm(range(iter)):
            optimizer.zero_grad()
            output = self.model(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            optimizer.step()
    
    def predict(self, X_star,device='cpu'):
        self.model.eval()
        self.likelihood.eval()
        X_star = torch.from_numpy(X_star).float().to(device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.model(X_star)
            pred_means = preds.mean.cpu().numpy()
            pred_vars = preds.variance.cpu().numpy()
        return pred_means, pred_vars
    
    def sample(self, X_star,n_sample,device='cpu'):
        self.model.eval()
        self.likelihood.eval()
        # X_star = self.X.numpy()
        pred_means, pred_vars = self.predict(X_star,device=device)
        scores = pred_means.flatten()
        variances = pred_vars.flatten()
        samples = scores + variances
        sorted_idx = torch.argsort(torch.tensor(samples), descending=True)
        top_samples_idx = sorted_idx[:n_sample]
        return X_star[top_samples_idx]



def test_fit_predict():
    # Generate some toy data
    np.random.seed(42)
    X = np.random.randn(10, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) #+ 0.01 * np.random.randn(10)
    X_star = np.random.randn(5, 2)
    Y_star = np.sin(X_star[:, 0]) + np.cos(X_star[:, 1])
    # Fit the Gaussian process regressor to the toy data
    gp = GaussianProcessActiveLearner(X, y)
    gp.fit(1000)

    # Make predictions and check that they're reasonable
    pred_means, pred_vars = gp.predict(X_star)
    assert pred_means.shape == (5,)
    assert pred_vars.shape == (5,)
    assert np.allclose(pred_means, Y_star, rtol=1e-3),f'{pred_means},{Y_star}'
    # assert np.allclose(pred_vars, np.array([0.1195, 0.1397, 0.1375, 0.1201, 0.1552]), rtol=1e-3)

def test_sample():
    # Generate some toy data
    np.random.seed(42)
    X = np.random.randn(10, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(10)

    # Fit the Gaussian process regressor to the toy data
    gp = GaussianProcessActiveLearner(X, y)
    gp.fit(500)

    # Sample some points from the Gaussian process and check that they're reasonable
    samples = gp.sample(3)
    assert samples.shape == (3, 2)
    assert np.allclose(samples, np.array([[ 0.4967, -0.1383], [ 1.5230, -0.2342], [-0.2342, -0.4695]]), rtol=1e-3)

if __name__ == '__main__':
    test_fit_predict()
    test_sample()
    pass