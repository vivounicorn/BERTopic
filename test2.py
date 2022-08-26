import utils.mkf_internal as mkf
import matplotlib.pyplot as plt


mkf.plot_3d_covariance((2, 3), [[1., 0], [0, 2.]])
#
mu = [2.0, 3.0]
P = [[1., 0.], [0., 2.]]
mkf.plot_3d_sampled_covariance(mu, P)
plt.show()